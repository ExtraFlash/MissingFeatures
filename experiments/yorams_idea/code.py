# model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
import configs.config
from torch.nn import functional as F


class MicrobeModel(pl.LightningModule):
    def __init__(
            self,
            num_microbes_train,
            num_samples,
            config: configs.config.Config,
            pretrained_embedding=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 0) Microbe embeddings (sample-specific)
        self.num_microbes = num_microbes_train
        self.num_samples = num_samples  # Training samples; test samples will append to this

        if isinstance(pretrained_embedding, nn.Embedding):
            base_embedding = pretrained_embedding.weight.detach().clone()
        elif isinstance(pretrained_embedding, torch.Tensor):
            base_embedding = pretrained_embedding.clone()
        else:
            base_embedding = torch.randn(self.num_microbes, config.EMBEDDING_DIM)

        self.microbe_embedding = nn.Parameter(
            base_embedding.unsqueeze(0).repeat(self.num_samples, 1, 1)
        )

        # 1) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.NHEAD_CLASSIFIER,
            dim_feedforward=4 * config.EMBEDDING_DIM,
            dropout=config.DROPOUT_LINEAR_CLASSIFIER,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS_CLASSIFIER)

        # 2) Learnable sample-specific W_k (will grow during test)
        self.W = nn.Parameter(torch.randn(self.num_samples, config.EMBEDDING_DIM))

        # 3) Classification head
        self.classifier = nn.Linear(config.EMBEDDING_DIM, 1)

        # 4) Hyperparams
        self.lr = config.LR_CLASSIFIER
        self.weight_decay = config.WEIGHT_DECAY_CLASSIFIER
        self.label_smoothing = config.LABEL_SMOOTHING

        # Loss functions
        self.reg_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        # Track test-time sample index
        self.test_sample_counter = self.num_samples

    def forward(self, sample_ids):
        Z_k = self.microbe_embedding[sample_ids]
        Z_prime = self.encoder(Z_k)  # Z'_ik: (batch_size, num_microbes, embedding_dim)
        W_k = self.W[sample_ids]
        abundance_hat = torch.einsum('bnd,bd->bn', Z_prime, W_k)
        cls_output = self.classifier(W_k)
        return abundance_hat, cls_output, Z_prime

    def _common_step(self, batch, batch_idx, mode):
        abundance_true, labels, which_dataset, sample_ids = batch
        abundance_hat, cls_logits, Z_prime = self(sample_ids)

        reg_loss = self.reg_loss_fn(abundance_hat, abundance_true)
        cls_loss = self._compute_loss(cls_logits, labels)
        total_loss = reg_loss + cls_loss

        metric = {'train': self.train_auroc, 'val': self.val_auroc, 'test': self.test_auroc}[mode]
        metric.update(cls_logits.sigmoid(), labels.long())

        self.log(f'{mode}_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(f'{mode}_reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(f'{mode}_cls_loss', cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.fine_tune_WK(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.fine_tune_WK(batch, batch_idx, 'test')

    def fine_tune_WK(self, batch,batch_idx,mode):
        """
        For a new sample, fine-tune W_k with frozen Transformer, then classify.
        """
        abundance_true, labels, which_dataset, sample_ids = batch
        sample_ids= sample_ids.to(self.device)
        if mode =='test':
            c=0
        # Freeze Transformer and embeddings
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.microbe_embedding.requires_grad_(False)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False



        # Compute Z_prime once with frozen encoder
        with torch.no_grad():
            Z_k = self.microbe_embedding[sample_ids]
            Z_prime = self.encoder(Z_k) # (batch_size, num_microbes, embedding_dim)

        # Fine-tune W_k for regression
        optimizer = torch.optim.AdamW([self.W], lr=self.lr, weight_decay=self.weight_decay)
        num_steps = 1  # Adjust as needed
        for _ in range(num_steps):
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            W_k = self.W[sample_ids]
            abundance_hat = torch.einsum('bnd,bd->bn', Z_prime,W_k)
            reg_loss = self.reg_loss_fn(abundance_hat, abundance_true)
            reg_loss.backward()  # Gradients flow to self.W[sample_ids]
            optimizer.step()

        # Compute metrics with optimized W_k
        with torch.no_grad():
            W_k = self.W[sample_ids]
            cls_logits = self.classifier(W_k).squeeze(-1)
            metric = {'train': self.train_auroc, 'val': self.val_auroc, 'test': self.test_auroc}[mode]
            metric.update(cls_logits.sigmoid(), labels.long())
            self.log(f'{mode}_loss', reg_loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log(f'{mode}_cls_logits', cls_logits.mean(), on_step=True, on_epoch=True)

        # Unfreeze for next batch or future training
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
        self.microbe_embedding.requires_grad_(True)
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.classifier.train()

    def on_train_epoch_end(self):
        auc_train = self.train_auroc.compute()
        self.log("train_auc_tag", auc_train, prog_bar=True, on_step=False, on_epoch=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        auc_val = self.val_auroc.compute()
        self.log("val_auc_tag", auc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        auc_test = self.test_auroc.compute()
        self.log("test_auc_tag", auc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.test_auroc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def _compute_loss(self, logits, labels):
        if self.label_smoothing > 0.0:
            smoothed_labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            return F.binary_cross_entropy_with_logits(logits, smoothed_labels)
        return F.binary_cross_entropy_with_logits(logits, labels.float())
