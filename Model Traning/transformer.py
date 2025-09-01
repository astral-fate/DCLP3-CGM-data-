import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import gc ## MODIFICATION: Import the garbage collection module

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer to understand time sequences."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadGlucoseAttention(nn.Module):
    """Optimized multi-head attention for faster training."""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        return output, attention_weights

class GlucoseTransformerBlock(nn.Module):
    """Optimized transformer block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadGlucoseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attention_weights

class OptimizedGlucosePredictor(pl.LightningModule):
    """Optimized Glucose Predictor for faster training on 9M+ samples."""
    def __init__(self,
                 input_features=12,
                 d_model=128,
                 n_heads=8,
                 n_layers=4,
                 d_ff=512,
                 max_seq_length=72,
                 n_patients=168,
                 prediction_horizon=6,
                 dropout=0.1,
                 learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.input_projection = nn.Linear(input_features, d_model)
        self.patient_embedding = nn.Embedding(n_patients, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.transformer_layers = nn.ModuleList([
            GlucoseTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.glucose_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.trend_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3)
        )
        self.risk_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2)
        )
        self.dropout = nn.Dropout(dropout)
        self.glucose_weight = 1.0
        self.trend_weight = 0.2
        self.risk_weight = 0.3

    def forward(self, x, patient_ids, mask=None):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        patient_emb = self.patient_embedding(patient_ids)
        patient_emb = patient_emb.expand(-1, seq_len, -1)
        x = x + patient_emb
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        for transformer_layer in self.transformer_layers:
            x, _ = transformer_layer(x, mask)
        final_representation = x[:, -1, :]
        glucose_pred = self.glucose_predictor(final_representation).squeeze(-1)
        trend_pred = self.trend_classifier(final_representation)
        risk_pred = self.risk_predictor(final_representation)
        return {'glucose': glucose_pred, 'trend': trend_pred, 'risk': risk_pred}

    def clinical_loss(self, predictions, targets):
        """Simplified clinical loss for faster computation."""
        glucose_pred = predictions['glucose']
        glucose_target = targets['glucose'].squeeze()
        weights = torch.ones_like(glucose_target)
        weights[glucose_target < 70] = 3.0
        weights[glucose_target > 250] = 2.0
        mse_loss = torch.mean((glucose_pred - glucose_target) ** 2 * weights)
        trend_loss = 0
        if 'trend' in targets:
            trend_target = targets['trend'].squeeze().long()
            trend_loss = nn.CrossEntropyLoss()(predictions['trend'], trend_target)
        risk_loss = 0
        if 'risk' in targets:
            risk_loss = nn.BCEWithLogitsLoss()(predictions['risk'], targets['risk'])
        total_loss = (self.glucose_weight * mse_loss +
                      self.trend_weight * trend_loss +
                      self.risk_weight * risk_loss)
        return total_loss, {
            'mse': mse_loss.item(),
            'trend_loss': trend_loss.item() if isinstance(trend_loss, torch.Tensor) else 0,
            'risk_loss': risk_loss.item() if isinstance(risk_loss, torch.Tensor) else 0
        }

    def training_step(self, batch, batch_idx):
        predictions = self(batch['features'], batch['patient_ids'])
        loss, loss_components = self.clinical_loss(predictions, batch['targets'])
        if batch_idx % 100 == 0:
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mse', loss_components['mse'])
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch['features'], batch['patient_ids'])
        loss, loss_components = self.clinical_loss(predictions, batch['targets'])
        glucose_pred = predictions['glucose']
        glucose_target = batch['targets']['glucose'].squeeze()
        mard = torch.mean(torch.abs(glucose_pred - glucose_target) / (glucose_target + 1e-6)) * 100
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mard', mard, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class FastCGMDataset(Dataset):
    """Optimized dataset class for faster data loading."""
    def __init__(self, df, sequence_length=72, prediction_horizon=6, patient_col='PtID'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.patient_col = patient_col
        print("Optimizing dataset for fast loading...")
        self.df = df.sort_values([patient_col, 'DataDtTm']).reset_index(drop=True)
        unique_patients = df[patient_col].unique()
        self.patient_to_id = {patient: idx for idx, patient in enumerate(unique_patients)}
        self.feature_cols = [col for col in df.columns
                             if col not in [patient_col, 'DataDtTm', 'glucose_target', 'Period', 'RecID', 'DataDtTm_adjusted', 'BolusType']]
        self.valid_indices = self._compute_valid_indices_fast()
        self.feature_data = self.df[self.feature_cols].astype(np.float32).fillna(0).values
        self.glucose_targets = self.df['glucose_target'].values
        self.cgm_values = self.df['CGM'].values
        self.patient_ids = self.df[self.patient_col].map(self.patient_to_id).values
        
        ## MODIFICATION: Delete the dataframe attribute after caching data into numpy arrays
        del self.df
        gc.collect()

        print(f"Fast dataset initialized: {len(self.valid_indices)} samples, {len(self.feature_cols)} features")

    def _compute_valid_indices_fast(self):
        valid_indices = []
        grouped = self.df.groupby(self.patient_col)
        for patient, group in grouped:
            start_idx = group.index[0]
            end_idx = group.index[-1]
            valid_start = start_idx + self.sequence_length
            valid_end = end_idx - self.prediction_horizon + 1
            if valid_end > valid_start:
                patient_valid_indices = list(range(valid_start, valid_end))
                valid_indices.extend(patient_valid_indices)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        central_idx = self.valid_indices[idx]
        start_idx = central_idx - self.sequence_length
        sequence_data = self.feature_data[start_idx:central_idx]
        target_idx = central_idx + self.prediction_horizon
        glucose_target = self.glucose_targets[target_idx]
        current_glucose = self.cgm_values[central_idx]
        patient_id = self.patient_ids[central_idx]
        glucose_change = glucose_target - current_glucose
        if glucose_change < -5:
            trend_target = 0
        elif glucose_change > 5:
            trend_target = 2
        else:
            trend_target = 1
        hypo_risk = 1.0 if glucose_target < 70 else 0.0
        hyper_risk = 1.0 if glucose_target > 180 else 0.0
        return {
            'features': torch.FloatTensor(sequence_data),
            'patient_ids': torch.LongTensor([patient_id]),
            'targets': {
                'glucose': torch.FloatTensor([glucose_target]),
                'trend': torch.LongTensor([trend_target]),
                'risk': torch.FloatTensor([hypo_risk, hyper_risk])
            }
        }

## MODIFICATION: Reduced the default batch size from 128 to 64
def train_optimized_model(processed_filepath, batch_size=64, max_epochs=20):
    """Fast training function with optimizations."""
    print("Loading dataset with optimizations...")
    df = pd.read_parquet(processed_filepath)

    unique_patients = df['PtID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    train_patient_ids = unique_patients[:int(len(unique_patients) * 0.8)]
    val_patient_ids = unique_patients[int(len(unique_patients) * 0.8):]

    train_df = df[df['PtID'].isin(train_patient_ids)]
    val_df = df[df['PtID'].isin(val_patient_ids)]

    ## MODIFICATION: Delete the original large DataFrame to free up RAM
    del df
    gc.collect()

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    train_dataset = FastCGMDataset(train_df, sequence_length=72)
    val_dataset = FastCGMDataset(val_df, sequence_length=72)
    
    ## MODIFICATION: Delete the split DataFrames as they are no longer needed
    del train_df, val_df
    gc.collect()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        ## MODIFICATION: Reduced num_workers to 2 to conserve RAM
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        shuffle=False, 
        ## MODIFICATION: Reduced num_workers to 1 to conserve RAM
        num_workers=1, 
        pin_memory=True,
        persistent_workers=True
    )

    model = OptimizedGlucosePredictor(
        input_features=len(train_dataset.feature_cols),
        d_model=128,
        n_heads=8,
        n_layers=4,
        n_patients=len(unique_patients), # MODIFICATION: Correctly use all unique patients
        learning_rate=2e-4
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        val_check_interval=0.5,
        log_every_n_steps=50,
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_mard', 
                mode='min',
                save_top_k=2
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_mard', 
                patience=3,
                min_delta=0.1
            )
        ]
    )
    print("Starting optimized training...")
    print(f"Expected batches per epoch: {len(train_loader):,}")
    
    trainer.fit(model, train_loader, val_loader)
    return model, trainer

# You can now run the training function
model, trainer = train_optimized_model('/content/drive/MyDrive/A glucose monitor/output/processed_glucose_data.parquet')
