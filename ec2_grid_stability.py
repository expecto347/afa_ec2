import argparse
import time
import os
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal

class GridStabilityDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.features = df.iloc[:, :-1].values.astype(np.float32)
            
        self.labels = df.iloc[:, -1].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


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
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot

def multiclass_accuracy(pred_probs_or_logits, y):
    pred_labels = torch.argmax(pred_probs_or_logits, dim=1)
    return (pred_labels == y).float().mean()


class Predictor(nn.Module):
    def __init__(self, d_in, d_out, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        recon = self.decoder(z)
        return recon

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
    def __init__(self, predictor, mask_layer, lr, factor=0.5, patience=10, min_lr=1e-6):
        super().__init__()
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mask_size = mask_layer.mask_size
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        loss = self.loss_fn(pred, y)

        self.log("Loss Train/Total", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)
        loss = self.loss_fn(pred, y)
        acc = multiclass_accuracy(pred, y)

        self.log("Loss Val/Total", loss, prog_bar=True, logger=True)
        self.log("Acc Val/Predictor", acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

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
    def __init__(self, value_network, predictor, pvae, mask_layer, lr, max_features, eps, mc_samples=16, factor=0.5, patience=10, min_lr=1e-6, disable_early_stopping=False):
        super().__init__()
        self.value_network = value_network
        self.predictor = predictor
        self.pvae = pvae
        self.mask_layer = mask_layer
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.disable_early_stopping = disable_early_stopping

        self.max_features = max_features
        self.mask_size = self.mask_layer.mask_size
        self.feature_costs = torch.ones(self.mask_size)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.reconstruction_loss = GaussianNLLLoss(reduction='none', var_penalty_weight=0.01)

        self.eps = eps
        self.eps_decay = 0.2
        self.eps_steps = 5
        self.mc_samples = mc_samples
        self.early_stopping_epochs = patience + 1
        self.automatic_optimization = False

    def on_fit_start(self):
        self.num_epsilon_steps = 0
        self.num_bad_epochs = 0
        self.pvae.eval() 
        self.predictor.eval() 

    def on_train_epoch_start(self):
        self._train_step_outputs = []

    def on_validation_epoch_start(self):
        self._val_step_outputs = []

    def estimate_class_frequency_mc(self, x, mask, x_masked):
        with torch.no_grad():
            mean, logvar = self.pvae.encode(x_masked)
            std = torch.exp(0.5 * logvar)
            
            num_mc = self.mc_samples
            batch_size, _ = x.shape

            mean_expanded = mean.unsqueeze(0).expand(num_mc, -1, -1)
            std_expanded = std.unsqueeze(0).expand(num_mc, -1, -1)

            eps = torch.randn_like(mean_expanded)
            z = mean_expanded + eps * std_expanded
            z = z.reshape(num_mc * batch_size, -1)

            recon_mu = self.pvae.decode(z)
            sampled_features = recon_mu.reshape(num_mc, batch_size, -1)

            x_expanded = x.unsqueeze(0)
            mask_expanded = mask.unsqueeze(0)
            x_completed = mask_expanded * x_expanded + (1 - mask_expanded) * sampled_features
            x_completed = x_completed.reshape(num_mc * batch_size, -1)

            full_mask = torch.ones(num_mc * batch_size, self.mask_size, dtype=mask.dtype, device=mask.device)
            predictor_input = self.mask_layer(x_completed, full_mask)
            pred_main_samples = self.predictor(predictor_input)

            num_classes = pred_main_samples.shape[1]
            pred_labels = torch.argmax(pred_main_samples, dim=1)
            one_hot = nn.functional.one_hot(pred_labels, num_classes=num_classes).float()
            class_frequency = one_hot.reshape(num_mc, batch_size, num_classes).mean(dim=0)
        return class_frequency

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0

        x_masked = self.mask_layer(x, mask)
        with torch.no_grad():
            _, recon = self.pvae(x_masked)
            pred_aux_old_det = recon
        
        p_curr = self.estimate_class_frequency_mc(x, mask, x_masked)
        v_curr = 1 - torch.sum(p_curr ** 2, dim=1)

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            pred_ec2 = self.value_network(x_masked)

            best = torch.argmax(pred_ec2 / self.feature_costs.to(x.device), dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            
            with torch.no_grad():
                _, recon = self.pvae(x_masked)
                pred_aux_det = recon
            
            p_next = self.estimate_class_frequency_mc(x, mask, x_masked)
            v_next = 1 - torch.sum(p_next ** 2, dim=1)

            batch_indices = torch.arange(len(x), device=x.device)
            pred_v_i = pred_aux_old_det[batch_indices, actions] 
            v_i = x[batch_indices, actions]
            
            tau = 0.1
            w = torch.exp(- ((v_i - pred_v_i) ** 2) / (2 * tau))
            target = v_curr - (w ** 2) * v_next
            
            value_network_loss = nn.functional.mse_loss(pred_ec2[batch_indices, actions], target)
            self.manual_backward(torch.mean(value_network_loss) / self.max_features)
            value_network_loss_total += torch.mean(value_network_loss).detach()

            v_curr = v_next
            pred_aux_old_det = pred_aux_det

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

            pred_ec2 -= 1e6 * mask
            best_feature_index = torch.argmax(pred_ec2 / self.feature_costs.to(x.device), dim=1)
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
        pred_acc_ = [multiclass_accuracy(preds, y) for preds in preds_cat]

        pred_acc_mean = torch.stack(pred_acc_).mean()
        pred_acc_final = pred_acc_[-1]

        self.log("Acc Val/Predictor Mean", pred_acc_mean, prog_bar=True, logger=True)
        self.log("Acc Val/Predictor Final", pred_acc_final, prog_bar=True, logger=True)

        sch = self.lr_schedulers()
        if sch is not None:
            metric = pred_acc_mean 
            sch.step(metric)

            if metric == sch.best:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--data_dir", type=str, default="/data/users/haozhes/DIME/experiments/grid_stability/data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true", help="Only pretrain PVAE and predictor")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping")
    return parser.parse_args()

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

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    train_data = GridStabilityDataset(os.path.join(args.data_dir, "train.csv"))
    val_data = GridStabilityDataset(os.path.join(args.data_dir, "val.csv"))
    test_data = GridStabilityDataset(os.path.join(args.data_dir, "test.csv"))

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    d_in = 12
    d_out = 2
    hidden = 512
    latent_dim = 8
    dropout = 0.3

    predictor = Predictor(d_in, d_out, hidden, dropout)
    pvae = PVAE(d_in, hidden, latent_dim, dropout)
    mask_layer = MaskLayer(mask_size=d_in, append=True)

    value_network = nn.Sequential(
        nn.Linear(d_in * 2, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_in),
    )

    run_description = f"ec2_grid_stability_trial_0"

    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(predictor, mask_layer, lr=1e-3)
    pred_logger = TensorBoardLogger(args.log_dir, name=run_description, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu, precision=32)
    pred_trainer.fit(pred_pretrainer, train_dataloader, val_dataloader)
    
    print("Testing Predictor...")
    pred_trainer.test(pred_pretrainer, dataloaders=test_dataloader)

    print("Pretraining PVAE...")
    pvae_pretrainer = PVAEPretrainer(pvae, mask_layer, lr=1e-3)
    pvae_logger = TensorBoardLogger(args.log_dir, name=run_description, version="pretrain_pvae")
    pvae_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pvae_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pvae_trainer = build_trainer(args.pretrain_epochs, pvae_logger, pvae_callbacks, args.gpu, precision=32)
    pvae_trainer.fit(pvae_pretrainer, train_dataloader, val_dataloader)
    
    print("Testing PVAE...")
    pvae_trainer.test(pvae_pretrainer, dataloaders=test_dataloader)

    if args.pretrain_only:
        print("Pretraining completed. Exiting because --pretrain_only is set.")
        return

    print("Training EC2 Estimator...")
    ec2_estimator = EC2Estimator(
        value_network, predictor, pvae, mask_layer,
        lr=1e-3, min_lr=1e-6,
        max_features=args.max_features, eps=0.1,
        disable_early_stopping=args.disable_early_stopping,
    )
    
    logger = TensorBoardLogger(args.log_dir, name=run_description, version="ec2_train")
    checkpoint_callback_perf = ModelCheckpoint(
        save_top_k=1, monitor="Acc Val/Predictor Final", mode="max", filename="best_val_perf_model", verbose=False
    )
    
    train_trainer = build_trainer(args.train_epochs, logger, [checkpoint_callback_perf, TQDMProgressBar(refresh_rate=50)], args.gpu, precision=32)
    train_trainer.fit(ec2_estimator, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()