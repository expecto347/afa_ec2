import argparse
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from afabench.common.datasets.datasets import AFAContextDataset


# ==========================================
# 1. 异构损失函数 (Heterogeneous Loss)
# 分别处理离散 Context (CrossEntropy) 和连续 Block (GaussianNLL)
# ==========================================
class HeterogeneousLoss(nn.Module):
    def __init__(self, reduction='mean', var_penalty_weight=0.01, context_dim=5):
        super().__init__()
        self.reduction = reduction
        self.var_penalty_weight = var_penalty_weight
        self.context_dim = context_dim

    def forward(self, pred_aux, target):
        context_logits, mu_blocks, logvar_blocks = pred_aux
        
        # 1. Context Loss (交叉熵)
        target_context = target[:, :self.context_dim]
        # 将 One-hot 还原为索引
        target_context_idx = torch.argmax(target_context, dim=1) 
        ce_loss = nn.functional.cross_entropy(context_logits, target_context_idx, reduction='none')
        
        # 2. Block Loss (高斯 NLL)
        target_blocks = target[:, self.context_dim:]
        logvar_blocks = torch.clamp(logvar_blocks, min=-5.0, max=10.0)
        nll_loss = 0.5 * (torch.exp(-logvar_blocks) * (mu_blocks - target_blocks)**2 + logvar_blocks)
        var_penalty = torch.relu(-logvar_blocks) * self.var_penalty_weight
        block_loss = nll_loss + var_penalty # [Batch, 50]
        
        # 3. 组合 Loss
        # 将 Context 的单一 Loss 均分平铺到前 5 个维度上，以便保持 [Batch, 55] 的形状，兼容 EC2 算法
        ce_loss_expanded = ce_loss.unsqueeze(1).expand(-1, self.context_dim) / self.context_dim
        loss = torch.cat([ce_loss_expanded, block_loss], dim=1) # [Batch, 55]
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss


# ==========================================
# 2. 概率型表格 Transformer
# ==========================================
class TabularTransformerPredictor(nn.Module):
    def __init__(self, d_in=55, d_out=8, context_dim=5, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.d_in = d_in
        self.context_dim = context_dim
        
        # 表格分词器 (将 Value 和 Mask 标记投影为高维 Token)
        self.value_proj = nn.Linear(2, d_model)
        self.pos_embeddings = nn.Parameter(torch.randn(1, d_in, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer 核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True # 必须开启 batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 主任务头：接收 [CLS] token，进行 8 分类
        self.main_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_out)
        )
        
        # 异构辅助头
        # Context 头：输出 5 维 Logits
        self.context_head = nn.Linear(d_model, context_dim)
        # Block 头：输出 50 个维度的 (mu, logvar)
        self.block_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x_cat):
        batch_size = x_cat.size(0)
        x_masked = x_cat[:, :self.d_in]
        mask_flags = x_cat[:, self.d_in:]
        
        # [Batch, 55, 2]
        tokens_input = torch.stack([x_masked, mask_flags], dim=-1)
        x_emb = self.value_proj(tokens_input) + self.pos_embeddings
        
        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, x_emb], dim=1) # [Batch, 56, d_model]
        
        out_seq = self.transformer(x_seq)
        
        # 主任务
        main_out = self.main_head(out_seq[:, 0, :])
        
        # 辅助任务
        feat_out = out_seq[:, 1:, :] # 去掉 [CLS]，取后 55 个 Token
        
        # 将 Context 视作一个整体特征（平均前 5 个 token），输出分类 Logits
        context_tokens = feat_out[:, :self.context_dim, :]
        context_logits = self.context_head(context_tokens.mean(dim=1)) # [Batch, 5]
        
        # Block 视作 50 个连续特征，输出 mu 和 logvar
        block_tokens = feat_out[:, self.context_dim:, :]
        block_preds = self.block_head(block_tokens) # [Batch, 50, 2]
        mu_blocks = block_preds[:, :, 0]
        logvar_blocks = block_preds[:, :, 1]
        
        return main_out, (context_logits, mu_blocks, logvar_blocks)


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

def ind_to_onehot(inds, n):
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot

def multiclass_accuracy(pred_probs_or_logits, y):
    pred_labels = torch.argmax(pred_probs_or_logits, dim=1)
    return (pred_labels == y).float().mean()


# ==========================================
# 3. 针对异构输出调整的 EC2Estimator
# ==========================================
class EC2Estimator(pl.LightningModule):
    def __init__(self, value_network, predictor, mask_layer, lr, max_features, eps, loss_fn, val_loss_fn, aux_loss_fn=None, mc_samples=16, factor=0.2, patience=2, min_lr=1e-6, early_stopping_epochs=None, eps_decay=0.2, eps_steps=1, feature_costs=None):
        super().__init__()
        self.value_network = value_network
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs
        self.factor = factor
        self.max_features = max_features
        self.mask_size = self.mask_layer.mask_size
        self.set_feature_costs(feature_costs)
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_steps = eps_steps
        self.mc_samples = mc_samples
        self.automatic_optimization = False

    def set_feature_costs(self, feature_costs=None):
        if feature_costs is None:
            feature_costs = torch.ones(self.mask_size)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs)
        self.register_buffer("feature_costs", feature_costs)

    def on_fit_start(self):
        self.num_epsilon_steps = 0
        self.num_bad_epochs = 0

    def on_train_epoch_start(self):
        self._train_step_outputs = []

    def on_validation_epoch_start(self):
        self._val_step_outputs = []

    def get_value_network_loss_weight(self):
        return 1.0

    @staticmethod
    def _classification_accuracy(pred_probs_or_logits, y):
        pred_labels = torch.argmax(pred_probs_or_logits, dim=1)
        return (pred_labels == y).float().mean()

    @staticmethod
    def _get_mu_full(pred_aux_det):
        """将 Context概率 和 Block均值 缝合为 55 维的伪 mu，用于 EC2 计算奖励 (w)"""
        context_logits, mu_blocks, _ = pred_aux_det
        context_probs = torch.softmax(context_logits, dim=-1)
        return torch.cat([context_probs, mu_blocks], dim=1)

    @staticmethod
    def _aux_reconstruction_accuracy(pred_aux_det, x, context_dim=5):
        context_logits, mu_blocks, _ = pred_aux_det
        
        # Context Accuracy
        pred_ctx = torch.argmax(context_logits, dim=1)
        target_ctx = torch.argmax(x[:, :context_dim], dim=1)
        ctx_correct = (pred_ctx == target_ctx).float().unsqueeze(1).expand(-1, context_dim)
        
        # Block Accuracy
        tolerance = 0.1
        block_correct = ((mu_blocks - x[:, context_dim:]).abs() < tolerance).float()
        
        all_correct = torch.cat([ctx_correct, block_correct], dim=1)
        return all_correct.mean(), all_correct.all(dim=1).float().mean()

    def estimate_class_frequency_mc(self, x, mask, pred_aux_det, context_dim=5):
        with torch.no_grad():
            context_logits, mu_blocks, logvar_blocks = pred_aux_det
            num_mc = self.mc_samples
            batch_size = x.size(0)

            # 1. 对 Context 采用多项式采样 (Categorical)
            context_probs = torch.softmax(context_logits, dim=-1)
            flat_probs = context_probs.unsqueeze(0).expand(num_mc, -1, -1).reshape(-1, context_dim)
            sampled_ctx_idx = torch.multinomial(flat_probs, 1).squeeze(-1)
            sampled_ctx = nn.functional.one_hot(sampled_ctx_idx, num_classes=context_dim).float()
            sampled_ctx = sampled_ctx.reshape(num_mc, batch_size, context_dim)

            # 2. 对 Blocks 采用重参数化采样 (Gaussian)
            mean_blocks = mu_blocks.unsqueeze(0).expand(num_mc, -1, -1)
            std_blocks = torch.exp(0.5 * logvar_blocks).unsqueeze(0).expand(num_mc, -1, -1)
            noise = torch.randn_like(mean_blocks)
            sampled_blocks = mean_blocks + noise * std_blocks

            # 3. 缝合完整的特征样本
            sampled_features = torch.cat([sampled_ctx, sampled_blocks], dim=2)

            x_expanded = x.unsqueeze(0)
            mask_expanded = mask.unsqueeze(0)
            x_completed = mask_expanded * x_expanded + (1 - mask_expanded) * sampled_features
            x_completed = x_completed.reshape(num_mc * batch_size, -1)

            full_mask = torch.ones_like(x_completed)
            predictor_input = torch.cat([x_completed, full_mask], dim=1)

            pred_main_samples, _ = self.predictor(predictor_input)
            num_classes = pred_main_samples.shape[1]
            pred_labels = torch.argmax(pred_main_samples, dim=1)
            one_hot = nn.functional.one_hot(pred_labels, num_classes=num_classes).float()
            class_frequency = one_hot.reshape(num_mc, batch_size, num_classes).mean(dim=0)

        return class_frequency

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0; pred_loss_total = 0
        pred_acc_total = 0; aux_pixel_acc_total = 0; aux_image_acc_total = 0

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.predictor(x_masked)
        pred_aux_det = tuple(t.detach() for t in pred_aux)

        loss_main = self.loss_fn(pred_main, y)
        loss_aux = self.aux_loss_fn(pred_aux, x)
        pred_loss = loss_main.mean() + loss_aux.mean()

        pred_acc_total += self._classification_accuracy(pred_main.detach(), y).detach()
        aux_pixel_acc, aux_image_acc = self._aux_reconstruction_accuracy(pred_aux_det, x)
        aux_pixel_acc_total += aux_pixel_acc.detach()
        aux_image_acc_total += aux_image_acc.detach()

        pred_loss_total += pred_loss.detach()
        self.manual_backward(pred_loss / (self.max_features + 1))

        p_curr = self.estimate_class_frequency_mc(x, mask, pred_aux_det)
        v_curr = 1 - torch.sum(p_curr ** 2, dim=1)
        pred_aux_old_det = pred_aux_det

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            pred_ec2 = self.value_network(x_masked)

            best = torch.argmax(pred_ec2 / self.feature_costs, dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            pred_main, pred_aux = self.predictor(x_masked)
            pred_aux_det = tuple(t.detach() for t in pred_aux)

            loss_main = self.loss_fn(pred_main, y)
            loss_aux = self.aux_loss_fn(pred_aux, x)
            total_pred_loss = loss_main.mean() + loss_aux.mean()

            pred_acc_total += self._classification_accuracy(pred_main.detach(), y).detach()
            aux_pixel_acc, aux_image_acc = self._aux_reconstruction_accuracy(pred_aux_det, x)
            aux_pixel_acc_total += aux_pixel_acc.detach()
            aux_image_acc_total += aux_image_acc.detach()

            p_next = self.estimate_class_frequency_mc(x, mask, pred_aux_det)
            v_next = 1 - torch.sum(p_next ** 2, dim=1)

            batch_indices = torch.arange(len(x), device=x.device)
            
            # 使用 _get_mu_full 获取异构拼接后的预测均值
            pred_mu_full = self._get_mu_full(pred_aux_old_det)
            pred_v_i = pred_mu_full[batch_indices, actions] 
            v_i = x[batch_indices, actions]
            
            tau = 0.1
            w = torch.exp(- ((v_i - pred_v_i) ** 2) / (2 * tau))
            target = v_curr - (w ** 2) * v_next
            
            value_network_loss = nn.functional.mse_loss(pred_ec2[batch_indices, actions], target)
            total_loss = self.get_value_network_loss_weight() * torch.mean(value_network_loss) + total_pred_loss
            self.manual_backward(total_loss / (self.max_features + 1))

            value_network_loss_total += torch.mean(value_network_loss).detach()
            pred_loss_total += total_pred_loss.detach()
            v_curr = v_next
            pred_aux_old_det = pred_aux_det

        opt.step()
        out = {
            "value_network_loss": (value_network_loss_total / self.max_features).detach().cpu(),
            "predictor_loss": (pred_loss_total / (self.max_features + 1)).detach().cpu(),
            "predictor_acc": (pred_acc_total / (self.max_features + 1)).detach().cpu(),
            "aux_pixel_acc": (aux_pixel_acc_total / (self.max_features + 1)).detach().cpu(),
            "aux_image_acc": (aux_image_acc_total / (self.max_features + 1)).detach().cpu(),
        }
        self._train_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        outputs = self._train_step_outputs
        if len(outputs) == 0: return
        pred_loss = torch.stack([out["predictor_loss"] for out in outputs]).mean()
        value_network_loss = torch.stack([out["value_network_loss"] for out in outputs]).mean()
        pred_acc = torch.stack([out["predictor_acc"] for out in outputs]).mean()
        aux_pixel_acc = torch.stack([out["aux_pixel_acc"] for out in outputs]).mean()
        self.log("Loss Train/Mean", pred_loss, prog_bar=True)
        self.log("Value Loss Train/Mean", value_network_loss, prog_bar=True)
        self.log("Acc Train/Predictor Mean", pred_acc, prog_bar=True)
        self.log("Acc Train/Aux Pixel Mean", aux_pixel_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        _, pred_aux = self.predictor(x_masked)
        pred_aux_det = tuple(t.detach() for t in pred_aux)
        
        aux_pixel_acc, aux_image_acc = self._aux_reconstruction_accuracy(pred_aux_det, x)
        pred_list = [self.estimate_class_frequency_mc(x, mask, pred_aux_det)]

        for _ in range(self.max_features):
            x_masked = self.mask_layer(x, mask)
            pred_ec2 = self.value_network(x_masked)
            pred_ec2 -= 1e6 * mask
            best_feature_index = torch.argmax(pred_ec2 / self.feature_costs, dim=1)
            mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            x_masked = self.mask_layer(x, mask)
            _, pred_aux = self.predictor(x_masked)
            pred_aux_det = tuple(t.detach() for t in pred_aux)
            pred_list.append(self.estimate_class_frequency_mc(x, mask, pred_aux_det))

        out = {
            "pred_list": pred_list, "y": y,
            "aux_pixel_acc": aux_pixel_acc.detach(),
            "aux_image_acc": aux_image_acc.detach(),
        }
        self._val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_step_outputs
        if len(outputs) == 0: return

        pred_list = [out["pred_list"] for out in outputs]
        y_list = [out["y"] for out in outputs]
        y = torch.cat(y_list)
        preds_cat = [torch.cat(preds) for preds in zip(*pred_list)]
        pred_loss_ = [self.loss_fn(preds, y).mean() for preds in preds_cat]
        pred_acc_ = [self._classification_accuracy(preds, y) for preds in preds_cat]

        self.log("Loss Val/Mean", torch.stack(pred_loss_).mean(), prog_bar=True)
        self.log("Acc Val/Predictor Final", pred_acc_[-1], prog_bar=True)
        self.log("Acc Val/Aux Pixel Guess", torch.stack([out["aux_pixel_acc"] for out in outputs]).mean(), prog_bar=True)
        
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(torch.stack(pred_loss_).mean())
            if torch.stack(pred_loss_).mean() == sch.best: self.num_bad_epochs = 0
            else: self.num_bad_epochs += 1

            if self.num_bad_epochs > self.early_stopping_epochs:
                self.eps *= self.eps_decay
                self.num_bad_epochs = 0
                self.num_epsilon_steps += 1
                if self.num_epsilon_steps >= self.eps_steps: self.trainer.should_stop = True
                for g in self.optimizers().param_groups: g["lr"] = self.lr

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr)
        return {"optimizer": opt, "lr_scheduler": scheduler}


class WeightedEC2Estimator(EC2Estimator):
    def __init__(self, *args, lam=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
    def get_value_network_loss_weight(self): return self.lam

class MaskingPretrainerEC2(pl.LightningModule):
    def __init__(self, model, mask_layer, lr, loss_fn, val_loss_fn, aux_loss_fn, factor=0.2, patience=2, min_lr=1e-6, early_stopping_epochs=None):
        super().__init__()
        self.model = model; self.mask_layer = mask_layer; self.mask_size = self.mask_layer.mask_size
        self.lr = lr; self.factor = factor; self.patience = patience; self.min_lr = min_lr
        self.early_stopping_epochs = early_stopping_epochs if early_stopping_epochs is not None else patience + 1
        self.loss_fn = loss_fn; self.val_loss_fn = val_loss_fn; self.aux_loss_fn = aux_loss_fn

    def on_fit_start(self): self.num_bad_epochs = 0
    def on_validation_epoch_start(self): self._val_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)
        mask = (torch.rand(len(x), self.mask_size, device=x.device) > torch.rand(len(x), 1, device=x.device)).float()

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.model(x_masked)
        loss_main = self.loss_fn(pred_main, y)
        loss_aux = self.aux_loss_fn(pred_aux, x)
        loss_total = loss_main + loss_aux
        self.log("Loss Train/Total", loss_total, prog_bar=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)
        mask = (torch.rand(len(x), self.mask_size, device=x.device) > torch.rand(len(x), 1, device=x.device)).float()

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.model(x_masked)
        aux_loss = self.aux_loss_fn(pred_aux, x)
        pred_aux_det = tuple(t.detach() for t in pred_aux)
        aux_pixel_acc, aux_image_acc = EC2Estimator._aux_reconstruction_accuracy(pred_aux_det, x)
        out = {"pred_main": pred_main, "y": y, "aux_loss": aux_loss.detach(), "aux_pixel_acc": aux_pixel_acc.detach()}
        self._val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_step_outputs
        if len(outputs) == 0: return
        pred = torch.cat([out["pred_main"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        loss = self.loss_fn(pred, y)
        self.log("Loss Val", loss, prog_bar=True)
        self.log("Acc Val/Aux Pixel", torch.stack([out["aux_pixel_acc"] for out in outputs]).mean(), prog_bar=True)

        sch = self.lr_schedulers()
        if loss < sch.best: self.num_bad_epochs = 0
        else: self.num_bad_epochs += 1
        if self.num_bad_epochs > self.early_stopping_epochs: self.trainer.should_stop = True

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.factor, patience=self.patience, min_lr=self.min_lr)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "Loss Val"}


# ==========================================
# 4. 主函数及配置参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--mc_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=15)
    parser.add_argument("--data_dir", type=str, default="/tmp/mnist/")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true")
    return parser.parse_args()

def make_dataloaders(batch_size, seed, num_workers=8):
    full_dataset = AFAContextDataset(
        n_samples=20000, seed=seed,
        context_feature_std=0.0, informative_feature_std=0.0,
        non_informative_feature_mean=0.5, non_informative_feature_std=0.0,
        use_cheap_context_features=True, n_contexts=5
    )
    np.random.seed(seed)
    val_size = int(len(full_dataset) * 0.2)
    val_inds = np.sort(np.random.choice(len(full_dataset), size=val_size, replace=False))
    train_inds = np.setdiff1d(np.arange(len(full_dataset)), val_inds)

    train_dataloader = DataLoader(full_dataset.create_subset(train_inds), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(full_dataset.create_subset(val_inds), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_dataloader, val_dataloader, full_dataset

def build_trainer(max_epochs, logger, callbacks, gpu_index=0, precision=32):
    return Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=[gpu_index] if torch.cuda.is_available() else 1, max_epochs=max_epochs, precision=precision, logger=logger, num_sanity_val_steps=0, callbacks=callbacks)

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    d_in = 55; d_out = 8; hidden = 512; dropout = 0.3

    train_dataloader, val_dataloader, full_dataset = make_dataloaders(args.batch_size, args.seed, num_workers=args.num_workers)

    for trial in range(args.num_trials):
        start_time = time.time()
        run_description = f"ec2_max_features_{args.max_features}_trial_{trial}"

        # 核心替换：使用 Transformer 代替 TwoHeadMLP
        predictor = TabularTransformerPredictor(d_in=d_in, d_out=d_out, context_dim=5, d_model=128, nhead=4, num_layers=3, dropout=dropout)
        
        value_network = nn.Sequential(
            nn.Linear(d_in * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_in)
        )
        mask_layer = MaskLayer(mask_size=d_in, append=True)

        print("Pretraining predictor")
        pretrain = MaskingPretrainerEC2(
            predictor, mask_layer, lr=1e-3, loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=multiclass_accuracy,
            # 核心替换：使用异构的损失函数 (HeterogeneousLoss)
            aux_loss_fn=HeterogeneousLoss(reduction="mean", var_penalty_weight=0.01, context_dim=5),
        )

        pretrain_logger = TensorBoardLogger(args.log_dir, name=run_description, version="pretrain")
        # ========== 【新增：预训练模型的 Checkpoint 回调】 ==========
        pretrain_checkpoint = ModelCheckpoint(
            save_top_k=1,
            monitor="Loss Val",  # 监控 MaskingPretrainerEC2 中记录的验证集总 Loss
            mode="min",          # Loss 越小越好
            filename="best_pretrain_model",
            verbose=False,
        )

        pretrain_trainer = build_trainer(args.pretrain_epochs, pretrain_logger, [pretrain_checkpoint, TQDMProgressBar(refresh_rate=50)], args.gpu, 32)
        pretrain_trainer.fit(pretrain, train_dataloader, val_dataloader)

        if args.pretrain_only: continue

        print("Training EC2 estimator")
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        feature_costs = full_dataset.get_feature_acquisition_costs().to(device)
        
        ec2_estimator = WeightedEC2Estimator(
            value_network, predictor, mask_layer, lr=1e-3, min_lr=1e-6, eps_decay=0.2, max_features=args.max_features, eps=0.1,
            loss_fn=nn.CrossEntropyLoss(reduction="none"), val_loss_fn=multiclass_accuracy,
            # 同样替换：EC2 使用 none 的 HeterogeneousLoss
            aux_loss_fn=HeterogeneousLoss(reduction="none", var_penalty_weight=0.01, context_dim=5),
            mc_samples=args.mc_samples, eps_steps=5, patience=5, lam=args.lam, feature_costs=feature_costs,
        )

        logger = TensorBoardLogger(args.log_dir, name=run_description)
        checkpoint_callback_perf = ModelCheckpoint(save_top_k=1, monitor="Acc Val/Predictor Final", mode="max", filename="best_val_perf_model", verbose=False)

        train_trainer = build_trainer(args.train_epochs, logger, [checkpoint_callback_perf, TQDMProgressBar(refresh_rate=50)], args.gpu, 16)
        train_trainer.fit(ec2_estimator, train_dataloader, val_dataloader)
        print(f"Training time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()