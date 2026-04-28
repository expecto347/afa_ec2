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


# 【调整后】：自定义高斯负对数似然损失函数（移除不平衡加权）
class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction='mean', var_penalty_weight=0.01):
        """
        var_penalty_weight: 惩罚极小方差（极度自信）的权重。
        """
        super().__init__()
        self.reduction = reduction
        self.var_penalty_weight = var_penalty_weight

    def forward(self, pred_aux, target):
        mu, logvar = pred_aux
        
        # 将下限从 -10.0 提高到 -5.0，防止模型在初期预测出极小的方差（过度自信）
        logvar = torch.clamp(logvar, min=-5.0, max=10.0)
        
        # 基础 NLL 损失公式
        nll_loss = 0.5 * (torch.exp(-logvar) * (mu - target)**2 + logvar)
        
        # 方差惩罚（缓解过度自信）
        # 惩罚负数很大的 logvar（即方差趋近于 0）
        var_penalty = torch.relu(-logvar) * self.var_penalty_weight
        
        # 最终损失：直接使用基础的 nll_loss
        loss = nll_loss + var_penalty
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss # 保持 [Batch, Features] 维度用于 EC2 Estimator
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


def get_entropy(pred):
    return Categorical(logits=pred).entropy()


def get_confidence(pred):
    return torch.max(pred.softmax(dim=1), dim=1).values


def ind_to_onehot(inds, n):
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot


def multiclass_accuracy(pred_probs_or_logits, y):
    pred_labels = torch.argmax(pred_probs_or_logits, dim=1)
    return (pred_labels == y).float().mean()


class TwoHeadMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden, dropout):
        super().__init__()
        
        # 【修改点】：网络 f (主分类任务) 的独立主干，保持简单的两层
        self.main_backbone = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(hidden, d_out)

        # 【修改点】：网络 g (辅助重建和不确定性任务) 的独立主干，增加深度和宽度
        aux_hidden = hidden * 2  # 加宽至 1024
        self.aux_backbone = nn.Sequential(
            nn.Linear(d_in * 2, aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, hidden), # 增加第三层
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.aux_head = nn.Linear(hidden, d_in * 2)

    def forward(self, x):
        # 独立的特征提取
        h_main = self.main_backbone(x)
        main_out = self.main_head(h_main)

        h_aux = self.aux_backbone(x)
        aux_out = self.aux_head(h_aux)
        
        mu, logvar = torch.chunk(aux_out, 2, dim=-1)
        return main_out, (mu, logvar)


class EC2Estimator(pl.LightningModule):
    def __init__(
        self,
        value_network,
        predictor,
        mask_layer,
        lr,
        max_features,
        eps,
        loss_fn,
        val_loss_fn,
        aux_loss_fn=None,
        mc_samples=16,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        early_stopping_epochs=None,
        eps_decay=0.2,
        eps_steps=1,
        feature_costs=None,
    ):
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
    def _aux_reconstruction_accuracy(pred_aux_det, x):
        # 【修改点】：只用均值(mu)来评估重建精度
        mu, _ = pred_aux_det
        tolerance = 0.1
        pixel_acc = ((mu - x).abs() < tolerance).float().mean()
        image_acc = ((mu - x).abs() < tolerance).all(dim=1).float().mean()
        return pixel_acc, image_acc

    def estimate_class_frequency_mc(self, x, mask, pred_aux_det):
        with torch.no_grad():
            mu, logvar = pred_aux_det
            num_mc = self.mc_samples
            batch_size, num_features = x.shape

            # 【修改点】：真正的蒙特卡洛重参数化采样
            mean_features = mu.unsqueeze(0).expand(num_mc, -1, -1)
            # 计算标准差: std = exp(0.5 * logvar)
            std_features = torch.exp(0.5 * logvar).unsqueeze(0).expand(num_mc, -1, -1)
            
            # 采样: z = mu + std * epsilon (其中 epsilon ~ N(0, 1))
            noise = torch.randn_like(mean_features)
            sampled_features = mean_features + noise * std_features

            x_expanded = x.unsqueeze(0)
            mask_expanded = mask.unsqueeze(0)
            x_completed = mask_expanded * x_expanded + (1 - mask_expanded) * sampled_features
            x_completed = x_completed.reshape(num_mc * batch_size, num_features)

            full_mask = torch.ones(
                num_mc * batch_size,
                self.mask_size,
                dtype=mask.dtype,
                device=mask.device,
            )

            if getattr(self.mask_layer, "append", False):
                predictor_input = torch.cat([x_completed, full_mask], dim=1)
            else:
                predictor_input = self.mask_layer(x_completed, full_mask)

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
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0
        pred_acc_total = 0
        aux_pixel_acc_total = 0
        aux_image_acc_total = 0

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.predictor(x_masked)
        
        # 【修改点】：正确分离 detached tuple
        pred_aux_det = (pred_aux[0].detach(), pred_aux[1].detach())

        loss_main = self.loss_fn(pred_main, y)
        if self.aux_loss_fn is not None:
            loss_aux = self.aux_loss_fn(pred_aux, x)
            pred_loss = loss_main.mean() + loss_aux.mean()
        else:
            pred_loss = loss_main.mean()

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
            pred_aux_det = (pred_aux[0].detach(), pred_aux[1].detach())

            loss_main = self.loss_fn(pred_main, y)
            if self.aux_loss_fn is not None:
                loss_aux = self.aux_loss_fn(pred_aux, x)
                total_pred_loss = loss_main.mean() + loss_aux.mean()
            else:
                total_pred_loss = loss_main.mean()

            pred_acc_total += self._classification_accuracy(pred_main.detach(), y).detach()
            aux_pixel_acc, aux_image_acc = self._aux_reconstruction_accuracy(pred_aux_det, x)
            aux_pixel_acc_total += aux_pixel_acc.detach()
            aux_image_acc_total += aux_image_acc.detach()

            p_next = self.estimate_class_frequency_mc(x, mask, pred_aux_det)
            v_next = 1 - torch.sum(p_next ** 2, dim=1)

            batch_indices = torch.arange(len(x), device=x.device)
            
            # 【修改点】：EC2 计算 w 时，提取历史预测均值 mu [索引为0]
            pred_v_i = pred_aux_old_det[0][batch_indices, actions] 
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
        if len(outputs) == 0:
            return

        pred_loss = torch.stack([out["predictor_loss"] for out in outputs]).mean()
        value_network_loss = torch.stack([out["value_network_loss"] for out in outputs]).mean()
        pred_acc = torch.stack([out["predictor_acc"] for out in outputs]).mean()
        aux_pixel_acc = torch.stack([out["aux_pixel_acc"] for out in outputs]).mean()
        aux_image_acc = torch.stack([out["aux_image_acc"] for out in outputs]).mean()

        self.log("Loss Train/Mean", pred_loss, prog_bar=True, logger=True)
        self.log("Value Loss Train/Mean", value_network_loss, prog_bar=True, logger=True)
        self.log("Acc Train/Predictor Mean", pred_acc, prog_bar=True, logger=True)
        self.log("Acc Train/Aux Pixel Mean", aux_pixel_acc, prog_bar=False, logger=True)
        self.log("Acc Train/Aux Image Mean", aux_image_acc, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        _, pred_aux = self.predictor(x_masked)
        pred_aux_det = (pred_aux[0].detach(), pred_aux[1].detach())
        
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
            pred_aux_det = (pred_aux[0].detach(), pred_aux[1].detach())
            pred_list.append(self.estimate_class_frequency_mc(x, mask, pred_aux_det))

        out = {
            "pred_list": pred_list,
            "y": y,
            "aux_pixel_acc": aux_pixel_acc.detach(),
            "aux_image_acc": aux_image_acc.detach(),
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
        pred_loss_ = [self.loss_fn(preds, y).mean() for preds in preds_cat]
        val_loss_ = [self.val_loss_fn(preds, y) for preds in preds_cat]
        pred_acc_ = [self._classification_accuracy(preds, y) for preds in preds_cat]

        pred_loss_mean = torch.stack(pred_loss_).mean()
        val_loss_mean = torch.stack(val_loss_).mean()
        pred_acc_mean = torch.stack(pred_acc_).mean()
        pred_acc_final = pred_acc_[-1]
        pred_loss_final = pred_loss_[-1]
        val_loss_final = val_loss_[-1]
        aux_pixel_acc_mean = torch.stack([out["aux_pixel_acc"] for out in outputs]).mean()
        aux_image_acc_mean = torch.stack([out["aux_image_acc"] for out in outputs]).mean()

        self.log("Loss Val/Mean", pred_loss_mean, prog_bar=True, logger=True)
        self.log("Perf Val/Mean", val_loss_mean, prog_bar=True, logger=True)
        self.log("Loss Val/Final", pred_loss_final, prog_bar=True, logger=True)
        self.log("Perf Val/Final", val_loss_final, prog_bar=True, logger=True)
        self.log("Acc Val/Predictor Mean", pred_acc_mean, prog_bar=True, logger=True)
        self.log("Acc Val/Predictor Final", pred_acc_final, prog_bar=True, logger=True)
        self.log("Acc Val/Aux Pixel Guess", aux_pixel_acc_mean, prog_bar=False, logger=True)
        self.log("Acc Val/Aux Image Guess", aux_image_acc_mean, prog_bar=True, logger=True)
        self.log("Eps Value", self.eps, prog_bar=True, logger=True)

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(pred_loss_mean)

            if pred_loss_mean == sch.best:
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.early_stopping_epochs:
                self.eps *= self.eps_decay
                self.num_bad_epochs = 0
                self.num_epsilon_steps += 1
                print(f"Decaying eps to {self.eps:.5f}, step = {self.num_epsilon_steps}")

                if self.num_epsilon_steps >= self.eps_steps:
                    self.trainer.should_stop = True

                for g in self.optimizers().param_groups:
                    g["lr"] = self.lr

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


class WeightedEC2Estimator(EC2Estimator):
    def __init__(self, *args, lam=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam

    def get_value_network_loss_weight(self):
        return self.lam


class MaskingPretrainerEC2(pl.LightningModule):
    def __init__(
        self,
        model,
        mask_layer,
        lr,
        loss_fn,
        val_loss_fn,
        aux_loss_fn,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        early_stopping_epochs=None,
    ):
        super().__init__()

        self.model = model
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size

        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs

        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn
        self.aux_loss_fn = aux_loss_fn

    def on_fit_start(self):
        self.num_bad_epochs = 0

    def on_validation_epoch_start(self):
        self._val_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.model(x_masked)
        loss_main = self.loss_fn(pred_main, y)
        loss_aux = self.aux_loss_fn(pred_aux, x)
        loss_total = loss_main + loss_aux

        self.log("Loss Train/Main", loss_main, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Loss Train/Aux", loss_aux, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("Loss Train/Total", loss_total, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)

        mask = torch.rand(len(x), self.mask_size, device=x.device)
        ref = torch.rand(len(x), 1, device=x.device)
        mask = (mask > ref).float()

        x_masked = self.mask_layer(x, mask)
        pred_main, pred_aux = self.model(x_masked)
        aux_loss = self.aux_loss_fn(pred_aux, x)
        
        pred_aux_det = (pred_aux[0].detach(), pred_aux[1].detach())
        aux_pixel_acc, aux_image_acc = EC2Estimator._aux_reconstruction_accuracy(pred_aux_det, x)
        out = {
            "pred_main": pred_main,
            "y": y,
            "aux_loss": aux_loss.detach(),
            "aux_pixel_acc": aux_pixel_acc.detach(),
            "aux_image_acc": aux_image_acc.detach(),
        }
        self._val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_step_outputs
        if len(outputs) == 0:
            return

        pred = torch.cat([out["pred_main"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        loss = self.loss_fn(pred, y)
        val_loss = self.val_loss_fn(pred, y)
        aux_loss = torch.stack([out["aux_loss"] for out in outputs]).mean()
        loss_total = loss + aux_loss
        aux_pixel_acc = torch.stack([out["aux_pixel_acc"] for out in outputs]).mean()
        aux_image_acc = torch.stack([out["aux_image_acc"] for out in outputs]).mean()

        self.log("Loss Val", loss, prog_bar=True, logger=True)
        self.log("Loss Val/Main", loss, prog_bar=False, logger=True)
        self.log("Perf Val", val_loss, prog_bar=True, logger=True)
        self.log("Loss Val/Aux", aux_loss, prog_bar=True, logger=True)
        self.log("Loss Val/Total", loss_total, prog_bar=False, logger=True)
        self.log("Acc Val/Aux Pixel Guess", aux_pixel_acc, prog_bar=True, logger=True)
        self.log("Acc Val/Aux Image Guess", aux_image_acc, prog_bar=True, logger=True)

        sch = self.lr_schedulers()
        if loss < sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            self.trainer.should_stop = True

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "Loss Val",
        }


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
    parser.add_argument("--pretrain_only", action="store_true", help="Only run predictor pretraining and skip EC2 estimator training")
    return parser.parse_args()


def make_dataloaders(batch_size, seed, num_workers=8):
    full_dataset = AFAContextDataset(
        n_samples=20000,
        seed=seed,
        context_feature_std=0.0,
        informative_feature_std=0.0,
        non_informative_feature_mean=0.5,
        non_informative_feature_std=0.0,
        use_cheap_context_features=True,
        n_contexts=5
    )

    np.random.seed(seed)
    val_size = int(len(full_dataset) * 0.2)
    val_inds = np.sort(np.random.choice(len(full_dataset), size=val_size, replace=False))
    train_inds = np.setdiff1d(np.arange(len(full_dataset)), val_inds)

    train_dataset = full_dataset.create_subset(train_inds)
    val_dataset = full_dataset.create_subset(val_inds)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    
    return train_dataloader, val_dataloader, full_dataset


def build_trainer(max_epochs, logger, callbacks, gpu_index=0, precision=32):
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_index]
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32

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

    d_in = 55
    d_out = 8
    hidden = 512
    dropout = 0.3

    train_dataloader, val_dataloader, full_dataset = make_dataloaders(
        args.batch_size,
        args.seed,
        num_workers=args.num_workers,
    )

    for trial in range(args.num_trials):
        start_time = time.time()
        run_description = f"ec2_max_features_{args.max_features}_trial_{trial}"

        predictor = TwoHeadMLP(d_in, d_out, hidden, dropout)
        value_network = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in),
        )
        mask_layer = MaskLayer(mask_size=d_in, append=True)

        print("Pretraining predictor")
        print("-" * 8)
        pretrain = MaskingPretrainerEC2(
            predictor,
            mask_layer,
            lr=1e-3,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=multiclass_accuracy,
            # 【调整】：移除 pos_weight=5.0
            aux_loss_fn=GaussianNLLLoss(reduction="mean", var_penalty_weight=0.01),
        )

        pretrain_logger = TensorBoardLogger(
            args.log_dir,
            name=run_description,
            version="pretrain",
        )
        pretrain_trainer = build_trainer(
            max_epochs=args.pretrain_epochs,
            logger=pretrain_logger,
            callbacks=[TQDMProgressBar(refresh_rate=50)],
            gpu_index=args.gpu,
            precision=32,
        )
        pretrain_trainer.fit(pretrain, train_dataloader, val_dataloader)

        if args.pretrain_only:
            training_time = time.time() - start_time
            print("Pretrain-only mode enabled, skipping EC2 estimator training")
            print(f"Pretraining time: {training_time:.2f}s")
            continue

        print("Training EC2 estimator")
        print("-" * 8)
        
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        feature_costs = full_dataset.get_feature_acquisition_costs().to(device)
        
        ec2_estimator = WeightedEC2Estimator(
            value_network,
            predictor,
            mask_layer,
            lr=1e-3,
            min_lr=1e-6,
            eps_decay=0.2,
            max_features=args.max_features,
            eps=0.1,
            loss_fn=nn.CrossEntropyLoss(reduction="none"),
            val_loss_fn=multiclass_accuracy,
            # 【调整】：移除 pos_weight=5.0
            aux_loss_fn=GaussianNLLLoss(reduction="none", var_penalty_weight=0.01),
            mc_samples=args.mc_samples,
            eps_steps=5,
            patience=5,
            lam=args.lam,
            feature_costs=feature_costs,
        )

        logger = TensorBoardLogger(args.log_dir, name=run_description)
        checkpoint_callback_perf = ModelCheckpoint(
            save_top_k=1,
            monitor="Perf Val/Final",
            mode="max",
            filename="best_val_perf_model",
            verbose=False,
        )

        train_trainer = build_trainer(
            max_epochs=args.train_epochs,
            logger=logger,
            callbacks=[checkpoint_callback_perf, TQDMProgressBar(refresh_rate=50)],
            gpu_index=args.gpu,
            precision=16,
        )
        train_trainer.fit(ec2_estimator, train_dataloader, val_dataloader)

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f}s")


if __name__ == "__main__":
    main()