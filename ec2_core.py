import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.metrics import roc_auc_score
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader


class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction='mean', var_penalty_weight=0.01):
        super().__init__()
        self.reduction = reduction
        self.var_penalty_weight = var_penalty_weight

    def forward(self, pred_aux, target):
        mu, logvar = pred_aux
        logvar = torch.clamp(logvar, min=-5.0, max=10.0)
        nll_loss = 0.5 * (torch.exp(-logvar) * (mu - target)**2 + logvar)
        var_penalty = torch.relu(-logvar) * self.var_penalty_weight
        loss = nll_loss + var_penalty

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        return loss


class MaskLayer(nn.Module):
    def __init__(self, mask_size, append=True):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, mask):
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


def get_confidence(pred):
    return torch.max(pred.softmax(dim=1), dim=1).values


def ind_to_onehot(inds, n):
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot.scatter_(1, inds.unsqueeze(1), 1.0)
    return onehot


def multiclass_accuracy(pred_probs_or_logits, y):
    pred_labels = torch.argmax(pred_probs_or_logits, dim=1)
    return (pred_labels == y).float().mean()


def compute_feature_thresholds(X_train, n_bins=10):
    """Compute percentile-based candidate split thresholds. Returns [d_in, n_bins - 1]."""
    q = torch.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1, device=X_train.device)
    thresholds = torch.quantile(X_train, q, dim=0)   # [n_bins - 1, d_in]
    return thresholds.T.contiguous()                 # [d_in, n_bins - 1]


def _ec2_scores_from_bin_counts(bin_counts, bin_totals, N):
    """Shared Gini + EC2 computation across multiple bins.
    bin_counts: [C, d_in, n_bins]
    bin_totals: [d_in, n_bins]
    Returns: [d_in]
    """
    p_bin_cls = bin_counts / (bin_totals[None, :, :] + 1e-8)  # [C, d_in, n_bins]
    gini_bin = 1.0 - p_bin_cls.pow(2).sum(0)                  # [d_in, n_bins]
    p_bin = bin_totals / N                                    # [d_in, n_bins]
    scores = (p_bin.pow(3) * gini_bin).sum(dim=-1)            # [d_in]
    return scores


def knn_ec2_target(x, mask, X_train, y_train, feat_thresholds, k=50,
                   global_ec2_cache=None, bin_oh_train=None):
    """
    EC2 targets for all features via kNN + multiple bin discretization.

    For each feature i, calculates purity reduction across its bins:
      Score = sum_b p_b^3 * Gini_b

    When mask is all-zero (step=0), uses the entire training set instead of k
    neighbors so that the target is the exact global statistic (zero variance).

    global_ec2_cache: precomputed [d_in] global EC2 scores (constant, avoids O(N) recomputation).
    bin_oh_train: precomputed [N, d_in, n_bins] one-hot bin assignments for X_train
                  (avoids per-call recomputation in the kNN path).

    Returns: [B, d_in] EC2 score per feature
    """
    B, d_in = x.shape[0], x.shape[1]
    num_classes = int(y_train.max().item()) + 1
    N = X_train.shape[0]
    n_bins = feat_thresholds.shape[1] + 1
    device = x.device

    with torch.no_grad():
        zero_mask_rows    = mask.sum(dim=1) == 0   # [B]
        nonzero_mask_rows = ~zero_mask_rows

        ec2_scores = torch.zeros(B, d_in, device=device)

        # --- Global path (step=0): use all N training samples ---
        if zero_mask_rows.any():
            if global_ec2_cache is not None:
                global_ec2 = global_ec2_cache
            else:
                bin_indices_g = (X_train[:, :, None] > feat_thresholds[None, :, :]).sum(dim=-1)
                bin_oh_g = torch.zeros(N, d_in, n_bins, device=device)
                bin_oh_g.scatter_(-1, bin_indices_g.unsqueeze(-1), 1.0)

                labels_oh_g  = torch.zeros(N, num_classes, device=device)
                labels_oh_g.scatter_(1, y_train.unsqueeze(1).long(), 1.0)
                
                bin_counts_g = torch.einsum('nc,ndp->cdp', labels_oh_g, bin_oh_g)
                bin_totals_g = bin_oh_g.sum(dim=0)   # [d_in, n_bins]
                global_ec2 = _ec2_scores_from_bin_counts(bin_counts_g, bin_totals_g, N)
                
            ec2_scores[zero_mask_rows] = global_ec2.unsqueeze(0).expand(
                zero_mask_rows.sum(), -1
            )

        # --- kNN path: use k neighbors for rows with observed features ---
        if nonzero_mask_rows.any():
            x_nz    = x[nonzero_mask_rows]
            mask_nz = mask[nonzero_mask_rows]
            B_nz    = x_nz.shape[0]

            diff    = (X_train[None, :, :] - x_nz[:, None, :]) * mask_nz[:, None, :]
            dist_sq = diff.pow(2).sum(dim=-1)
            _, indices = torch.topk(dist_sq, k, largest=False, dim=1)

            neighbor_labels   = y_train[indices]    # [B_nz, k]
            neighbor_features = X_train[indices]    # [B_nz, k, d_in]

            if bin_oh_train is not None:
                bin_oh = bin_oh_train[indices].float()  # [B_nz, k, d_in, n_bins]
            else:
                neighbor_bin_indices = (
                    neighbor_features[:, :, :, None] > feat_thresholds[None, None, :, :]
                ).sum(dim=-1)  # [B_nz, k, d_in]
                bin_oh = torch.zeros(B_nz, k, d_in, n_bins, device=device)
                bin_oh.scatter_(-1, neighbor_bin_indices.unsqueeze(-1), 1.0)

            labels_oh = torch.zeros(B_nz, k, num_classes, device=device)
            labels_oh.scatter_(2, neighbor_labels.unsqueeze(-1).long(), 1.0)

            bin_counts  = torch.einsum('bkc,bkdp->bcdp', labels_oh, bin_oh)
            bin_totals  = bin_oh.sum(dim=1)   # [B_nz, d_in, n_bins]

            p_bin_cls  = bin_counts  / (bin_totals.unsqueeze(1) + 1e-8)
            gini_bin  = 1.0 - p_bin_cls.pow(2).sum(dim=1)   # [B_nz, d_in, n_bins]
            p_bin  = bin_totals  / k
            scores  = (p_bin.pow(2) * gini_bin).sum(dim=-1)
            ec2_scores[nonzero_mask_rows] = scores

    return ec2_scores


class Predictor(nn.Module):
    def __init__(self, d_in, d_out, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


class PVAE(nn.Module):
    def __init__(self, d_in, hidden, latent_dim, dropout, num_samples=16):
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(d_in * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, d_in)
        )

    def encode(self, x_masked):
        latent = self.encoder(x_masked)
        mean = latent[:, :self.latent_dim]
        logvar = latent[:, self.latent_dim:]
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_masked):
        mean, logvar = self.encode(x_masked)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.shape[0], self.num_samples, mean.shape[1], device=mean.device)
        z = torch.unsqueeze(mean, 1) + eps * torch.unsqueeze(std, 1)

        z = z.reshape(-1, self.latent_dim)
        recon = self.decoder(z)

        mu = recon.reshape(-1, self.num_samples, recon.shape[-1]).mean(dim=1)
        return (mean, logvar), mu


class PVAEPretrainer(pl.LightningModule):
    def __init__(self, pvae, mask_layer, lr, factor=0.5, patience=10, min_lr=1e-6, beta=0.05):
        super().__init__()
        self.pvae = pvae
        self.mask_layer = mask_layer
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.beta = beta
        self.mask_size = mask_layer.mask_size
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        self.early_stopping_epochs = patience + 1

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        (latent_mean, latent_logvar), mu = self.pvae(x_masked)

        latent_std = torch.exp(0.5 * latent_logvar)
        kl = torch.distributions.kl_divergence(
            Normal(latent_mean, latent_std),
            Normal(0.0, 1.0)
        ).sum(dim=1).mean()

        mse = self.reconstruction_loss(mu, x)
        loss = mse + self.beta * kl

        self.log("Loss Train/Total", loss, prog_bar=True, logger=True)
        self.log("Loss Train/KL", kl, prog_bar=False, logger=True)
        self.log("Loss Train/MSE", mse, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        (latent_mean, latent_logvar), mu = self.pvae(x_masked)

        latent_std = torch.exp(0.5 * latent_logvar)
        kl = torch.distributions.kl_divergence(
            Normal(latent_mean, latent_std),
            Normal(0.0, 1.0)
        ).sum(dim=1).mean()

        mse = self.reconstruction_loss(mu, x)
        loss = mse + self.beta * kl

        tolerance = 0.1
        pixel_acc = ((mu - x).abs() < tolerance).float().mean()

        self.log("Loss Val/Total", loss, prog_bar=True, logger=True)
        self.log("Acc Val/Pixel", pixel_acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        (latent_mean, latent_logvar), mu = self.pvae(x_masked)

        latent_std = torch.exp(0.5 * latent_logvar)
        kl = torch.distributions.kl_divergence(
            Normal(latent_mean, latent_std),
            Normal(0.0, 1.0)
        ).sum(dim=1).mean()

        mse = self.reconstruction_loss(mu, x)
        loss = mse + self.beta * kl

        tolerance = 0.1
        pixel_acc = ((mu - x).abs() < tolerance).float().mean()

        self.log("Loss Test/Total", loss, prog_bar=True, logger=True)
        self.log("Acc Test/Pixel", pixel_acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "Loss Val/Total"}


class PredictorPretrainer(pl.LightningModule):
    def __init__(self, predictor, mask_layer, lr, factor=0.5, patience=10, min_lr=1e-6,
                 class_weights=None, mask_retention_max=1.0, mask_retention_mix=0.0):
        super().__init__()
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mask_size = mask_layer.mask_size
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # Defaults (max=1.0, mix=0.0) = current behavior: per-feature retention
        # ~ U(0,1) (mean 50%).  mask_retention_max < 1 biases masks toward the
        # low budgets EC2 evaluation actually visits (1-10 features observed);
        # mask_retention_mix is the fraction of masks kept at full range so the
        # predictor does not forget the high-budget regime.
        self.mask_retention_max = mask_retention_max
        self.mask_retention_mix = mask_retention_mix

    def _sample_mask(self, x):
        if self.mask_retention_max >= 1.0:
            mask = torch.rand(len(x), self.mask_size, device=x.device)
            ref = torch.rand(len(x), 1, device=x.device)
            return (mask > ref).float()
        full = torch.rand(len(x), 1, device=x.device) < self.mask_retention_mix
        ret = torch.rand(len(x), 1, device=x.device)
        ret = torch.where(full, ret, ret * self.mask_retention_max)
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        return (mask < ret).float()

    def training_step(self, batch, batch_idx):
        x, y = batch
        mask = self._sample_mask(x)

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        loss = self.loss_fn(pred, y)

        self.log("Loss Train/Total", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = self._sample_mask(x)

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        loss = self.loss_fn(pred, y)
        acc = multiclass_accuracy(pred, y)

        self.log("Loss Val/Total", loss, prog_bar=True, logger=True)
        self.log("Acc Val/Predictor", acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask = self._sample_mask(x)

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        loss = self.loss_fn(pred, y)
        acc = multiclass_accuracy(pred, y)

        self.log("Loss Test/Total", loss, prog_bar=True, logger=True)
        self.log("Acc Test/Predictor", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "Loss Val/Total"}


class EC2Estimator(pl.LightningModule):
    def __init__(self, value_network, predictor, mask_layer,
                 X_train, y_train, feat_thresholds,
                 feat_mean, feat_std,
                 lr, max_features, eps,
                 k=50,
                 knn_subsample=None,
                 factor=0.5, patience=10, min_lr=1e-6, disable_early_stopping=False,
                 random_first_step=False,
                 eps_decay=0.2, eps_steps=8,
                 val_monitor='accuracy',
                 feature_groups=None):
        super().__init__()
        self.value_network = value_network
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.register_buffer('X_train', X_train)
        self.register_buffer('y_train', y_train)
        self.register_buffer('feat_thresholds', feat_thresholds)
        self.register_buffer('feat_mean', torch.tensor(feat_mean, dtype=torch.float32))
        self.register_buffer('feat_std',  torch.tensor(feat_std,  dtype=torch.float32))
        self.k = k
        self.knn_subsample = knn_subsample
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.disable_early_stopping = disable_early_stopping
        # When True, the first acquisition in every training episode is drawn
        # uniformly at random (ignoring epsilon-greedy).  This ensures the value
        # network is trained on all possible starting states and prevents the
        # policy from collapsing to a fixed acquisition order.
        self.random_first_step = random_first_step

        self.max_features = max_features
        self.mask_size = self.mask_layer.mask_size
        self.feature_costs = torch.ones(self.mask_size)

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_steps = eps_steps
        self.early_stopping_epochs = patience + 1
        self.automatic_optimization = False
        self.val_monitor = val_monitor

        self.feature_groups = feature_groups
        if feature_groups is not None:
            self.n_groups = len(feature_groups)
            gcm = torch.zeros(self.n_groups, self.mask_size, dtype=torch.bool)
            for g, cols in enumerate(feature_groups):
                for c in cols:
                    gcm[g, c] = True
            self.register_buffer('group_col_matrix', gcm)
        else:
            self.n_groups = self.mask_size

    def _col_scores_to_group_scores(self, pred_ec2, mask):
        """Aggregate per-column EC2 scores to per-group scores (min over group columns)."""
        penalized = pred_ec2 + 1e6 * mask  # already-observed cols → +inf
        gcm = self.group_col_matrix  # bool [n_groups, d_in]
        scores = penalized.unsqueeze(1).expand(-1, self.n_groups, -1)  # [B, n_groups, d_in]
        non_group = (~gcm).unsqueeze(0).float() * 1e6
        group_scores = (scores + non_group).min(dim=2).values  # [B, n_groups]
        return group_scores

    def _apply_group_action(self, mask, group_actions):
        """Set all columns belonging to selected groups to observed."""
        selected_cols = self.group_col_matrix[group_actions].float()  # [B, d_in]
        return torch.max(mask, selected_cols)

    def on_fit_start(self):
        self.num_epsilon_steps = 0
        self.num_bad_epochs = 0
        self.predictor.eval()

        # Precompute constants for knn_ec2_target (X_train / y_train / feat_thresholds are fixed).
        num_classes = int(self.y_train.max().item()) + 1
        N = self.X_train.shape[0]
        n_bins = self.feat_thresholds.shape[1] + 1
        with torch.no_grad():
            bin_indices_g = (self.X_train[:, :, None] > self.feat_thresholds[None, :, :]).sum(dim=-1)
            bin_oh_g = torch.zeros(N, self.X_train.shape[1], n_bins, device=self.device)
            bin_oh_g.scatter_(-1, bin_indices_g.unsqueeze(-1), 1.0)
            
            labels_oh_g = torch.zeros(N, num_classes, device=self.device)
            labels_oh_g.scatter_(1, self.y_train.unsqueeze(1).long(), 1.0)
            bin_counts_g  = torch.einsum('nc,ndp->cdp', labels_oh_g, bin_oh_g)
            bin_totals_g  = bin_oh_g.sum(dim=0)
            global_ec2 = _ec2_scores_from_bin_counts(bin_counts_g, bin_totals_g, N)
        self.register_buffer('global_ec2_cache', global_ec2)           # [d_in]
        self.register_buffer('bin_oh_train', bin_oh_g.bool())          # [N, d_in, n_bins]

        # Subsample X_train / y_train for kNN distance computation to bound
        # the [B, N_knn, d_in] intermediate tensor when N is large.
        if self.knn_subsample is not None and self.knn_subsample < N:
            perm = torch.randperm(N, device=self.device)[:self.knn_subsample]
            self.register_buffer('X_train_knn', self.X_train[perm])
            self.register_buffer('y_train_knn', self.y_train[perm])
            self.register_buffer('bin_oh_train_knn', bin_oh_g[perm].bool())
        else:
            self.register_buffer('X_train_knn', self.X_train)
            self.register_buffer('y_train_knn', self.y_train)
            self.register_buffer('bin_oh_train_knn', bin_oh_g.bool())

    def on_train_epoch_start(self):
        self._train_step_outputs = []

    def on_validation_epoch_start(self):
        self._val_step_outputs = []

    def estimate_class_frequency_mc(self, x, mask, x_masked):
        with torch.no_grad():
            class_freq = self.predictor(x_masked).softmax(dim=1)
        return class_freq

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0

        for step in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            pred_ec2 = self.value_network(x_masked)  # [B, d_in]

            if self.feature_groups is not None:
                group_scores = self._col_scores_to_group_scores(pred_ec2, mask)
                best = torch.argmin(group_scores, dim=1)
                random_action = torch.randint(self.n_groups, (len(x),), device=x.device)
            else:
                penalized = pred_ec2 + 1e6 * mask
                best = torch.argmin(penalized, dim=1)
                random_action = torch.randint(self.mask_size, (len(x),), device=x.device)
            if self.random_first_step and step == 0:
                actions = random_action
            else:
                exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
                actions = exploit * best + (1 - exploit) * random_action

            targets = knn_ec2_target(
                x, mask, self.X_train_knn, self.y_train_knn, self.feat_thresholds, k=self.k,
                global_ec2_cache=self.global_ec2_cache,
                bin_oh_train=self.bin_oh_train_knn,
            )  # [B, d_in]

            unobserved = (mask == 0).float()  # [B, d_in]
            loss_matrix = (pred_ec2 - targets).pow(2) * unobserved
            value_network_loss = loss_matrix.sum() / (unobserved.sum() + 1e-8)

            self.manual_backward(value_network_loss / self.max_features)
            value_network_loss_total += value_network_loss.detach()

            if self.feature_groups is not None:
                mask = self._apply_group_action(mask, actions)
            else:
                mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

        opt.step()
        out = {
            "value_network_loss": (value_network_loss_total / self.max_features).detach().cpu(),
        }
        self._train_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        outputs = self._train_step_outputs
        if len(outputs) == 0:
            return
        value_network_loss = torch.stack([out["value_network_loss"] for out in outputs]).mean()
        self.log("Value Loss Train/Mean", value_network_loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)

        pred_list = [self.estimate_class_frequency_mc(x, mask, x_masked)]

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            pred_ec2 = self.value_network(x_masked)

            if self.feature_groups is not None:
                group_scores = self._col_scores_to_group_scores(pred_ec2, mask)
                best_group_index = torch.argmin(group_scores, dim=1)
                mask = self._apply_group_action(mask, best_group_index)
            else:
                pred_ec2 += 1e6 * mask
                best_feature_index = torch.argmin(pred_ec2, dim=1)
                mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred_list.append(self.estimate_class_frequency_mc(x, mask, x_masked))

        out = {
            "pred_list": pred_list,
            "y": y,
        }
        self._val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_step_outputs
        if len(outputs) == 0:
            return

        pred_list = [out["pred_list"] for out in outputs]
        y_list = [out["y"] for out in outputs]
        y = torch.cat(y_list)
        preds_cat = [torch.cat(preds) for preds in zip(*pred_list)]

        # Accuracy
        pred_acc_ = [multiclass_accuracy(preds, y) for preds in preds_cat]
        pred_acc_mean  = torch.stack(pred_acc_).mean()
        pred_acc_final = pred_acc_[-1]
        self.log("Acc Val/Predictor Mean",  pred_acc_mean,  prog_bar=True,  logger=True)
        self.log("Acc Val/Predictor Final", pred_acc_final, prog_bar=False, logger=True)

        # AUROC
        y_np   = y.cpu().numpy()
        d_out  = preds_cat[0].shape[1]
        auroc_per_step = []
        for preds in preds_cat:
            probs = preds.cpu().numpy()
            try:
                if d_out == 2:
                    auc = roc_auc_score(y_np, probs[:, 1])
                else:
                    auc = roc_auc_score(y_np, probs, multi_class='ovr', average='macro')
            except ValueError:
                auc = float('nan')
            auroc_per_step.append(auc)
        auroc_mean  = float(np.nanmean(auroc_per_step))
        auroc_final = auroc_per_step[-1]
        self.log("AUROC Val/Predictor Mean",  auroc_mean,  prog_bar=True,  logger=True)
        self.log("AUROC Val/Predictor Final", auroc_final, prog_bar=False, logger=True)

        primary_metric = torch.tensor(auroc_mean) if self.val_monitor == 'auroc' else pred_acc_mean

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(primary_metric)

            if primary_metric >= sch.best:
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.early_stopping_epochs:
                self.eps *= self.eps_decay
                self.num_bad_epochs = 0
                self.num_epsilon_steps += 1
                if self.num_epsilon_steps >= self.eps_steps:
                    if not getattr(self, "disable_early_stopping", False):
                        self.trainer.should_stop = True

                for g in self.optimizers().param_groups:
                    g["lr"] = self.lr

    def configure_optimizers(self):
        opt = optim.Adam(self.value_network.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr, mode='max'
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


def build_trainer(max_epochs, logger, callbacks, gpu_index=0, precision=32):
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_index]
    else:
        accelerator = "cpu"
        devices = 1

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
