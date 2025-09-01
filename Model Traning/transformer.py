import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math

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
    """
    Optimized multi-head attention for faster training.
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Remove bias for speed
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Multi-head projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with Flash Attention pattern
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
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

        # Optimized feed forward with smaller intermediate size
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x, attention_weights

class OptimizedGlucosePredictor(pl.LightningModule):
    """
    Optimized Glucose Predictor for faster training on 9M+ samples.
    
    Key optimizations:
    - Smaller model size
    - Shorter sequences
    - Simplified architecture
    - Better GPU utilization
    """

    def __init__(self,
                 input_features=12,
                 d_model=128,  # Reduced from 256
                 n_heads=8,
                 n_layers=4,   # Reduced from 6
                 d_ff=512,     # Reduced from 1024
                 max_seq_length=72,  # Reduced from 288 (6 hours instead of 24)
                 n_patients=168,
                 prediction_horizon=6,
                 dropout=0.1,
                 learning_rate=2e-4):  # Increased learning rate
        super().__init__()

        self.save_hyperparameters()

        self.d_model = d_model
        self.prediction_horizon = prediction_horizon

        # Input projections
        self.input_projection = nn.Linear(input_features, d_model)
        self.patient_embedding = nn.Embedding(n_patients, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GlucoseTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Simplified output heads
        self.glucose_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Single step prediction for speed
        )

        # Simplified trend classifier
        self.trend_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3)  # 3 categories: fall, stable, rise
        )

        # Risk predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2)  # P(hypo), P(hyper)
        )

        self.dropout = nn.Dropout(dropout)

        # Loss weights
        self.glucose_weight = 1.0
        self.trend_weight = 0.2  # Reduced weight
        self.risk_weight = 0.3   # Reduced weight

    def forward(self, x, patient_ids, mask=None):
        batch_size, seq_len, _ = x.shape

        # Input embedding and positional encoding
        x = self.input_projection(x)

        # Add patient-specific embeddings
        patient_emb = self.patient_embedding(patient_ids)  # [batch, 1, d_model]
        patient_emb = patient_emb.expand(-1, seq_len, -1)  # [batch, seq, d_model]
        x = x + patient_emb

        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)

        x = self.dropout(x)

        # Pass through transformer layers
        for transformer_layer in self.transformer_layers:
            x, _ = transformer_layer(x, mask)

        # Use the last timestep for prediction
        final_representation = x[:, -1, :]

        # Multi-task outputs
        glucose_pred = self.glucose_predictor(final_representation).squeeze(-1)  # [batch]
        trend_pred = self.trend_classifier(final_representation)     # [batch, 3]
        risk_pred = self.risk_predictor(final_representation)        # [batch, 2]

        return {
            'glucose': glucose_pred,
            'trend': trend_pred,
            'risk': risk_pred
        }

    def clinical_loss(self, predictions, targets):
        """
        Simplified clinical loss for faster computation.
        """
        glucose_pred = predictions['glucose']
        glucose_target = targets['glucose'].squeeze()

        # Base MSE loss with clinical weighting
        error = torch.abs(glucose_pred - glucose_target)
        
        # Apply clinical weights in a vectorized way
        weights = torch.ones_like(glucose_target)
        weights[glucose_target < 70] = 3.0      # Hypoglycemia
        weights[glucose_target > 250] = 2.0     # Severe hyperglycemia
        
        mse_loss = torch.mean((glucose_pred - glucose_target) ** 2 * weights)

        # Trend classification loss
        trend_loss = 0
        if 'trend' in targets:
            trend_target = targets['trend'].squeeze().long()
            trend_loss = nn.CrossEntropyLoss()(predictions['trend'], trend_target)

        # Risk prediction loss
        risk_loss = 0
        if 'risk' in targets:
            risk_loss = nn.BCEWithLogitsLoss()(predictions['risk'], targets['risk'])

        # Combined loss
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

        # Log only essential metrics
        if batch_idx % 100 == 0:  # Log less frequently
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mse', loss_components['mse'])

        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch['features'], batch['patient_ids'])
        loss, loss_components = self.clinical_loss(predictions, batch['targets'])

        # Clinical metrics
        glucose_pred = predictions['glucose']
        glucose_target = batch['targets']['glucose'].squeeze()

        # MARD calculation
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

        # Simpler scheduler
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
    """
    Optimized dataset class for faster data loading.
    """

    def __init__(self, df, sequence_length=72, prediction_horizon=6, patient_col='PtID'):
        self.sequence_length = sequence_length  # 6 hours instead of 12
        self.prediction_horizon = prediction_horizon
        self.patient_col = patient_col

        print("Optimizing dataset for fast loading...")
        
        # Sort and reset index
        self.df = df.sort_values([patient_col, 'DataDtTm']).reset_index(drop=True)
        
        # Create patient mapping
        unique_patients = df[patient_col].unique()
        self.patient_to_id = {patient: idx for idx, patient in enumerate(unique_patients)}

        # Feature columns
        self.feature_cols = [col for col in df.columns
                           if col not in [patient_col, 'DataDtTm', 'glucose_target', 'Period', 'RecID', 'DataDtTm_adjusted', 'BolusType']]
        
        # Pre-compute valid indices more efficiently
        self.valid_indices = self._compute_valid_indices_fast()
        
        # Pre-load feature data for faster access
        self.feature_data = self.df[self.feature_cols].astype(np.float32).fillna(0).values
        self.glucose_targets = self.df['glucose_target'].values
        self.cgm_values = self.df['CGM'].values
        self.patient_ids = self.df[patient_col].map(self.patient_to_id).values

        print(f"Fast dataset initialized: {len(self.valid_indices)} samples, {len(self.feature_cols)} features")

    def _compute_valid_indices_fast(self):
        """Faster way to compute valid indices."""
        valid_indices = []
        
        # Group by patient for efficient processing
        grouped = self.df.groupby(self.patient_col)
        
        for patient, group in grouped:
            start_idx = group.index[0]
            end_idx = group.index[-1]
            
            # Vectorized computation of valid indices
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

        # Extract sequence using pre-loaded arrays
        start_idx = central_idx - self.sequence_length
        sequence_data = self.feature_data[start_idx:central_idx]

        # Extract targets
        target_idx = central_idx + self.prediction_horizon
        glucose_target = self.glucose_targets[target_idx]
        current_glucose = self.cgm_values[central_idx]
        patient_id = self.patient_ids[central_idx]

        # Simplified trend classification (3 categories instead of 5)
        glucose_change = glucose_target - current_glucose
        if glucose_change < -5:
            trend_target = 0  # Fall
        elif glucose_change > 5:
            trend_target = 2  # Rise  
        else:
            trend_target = 1  # Stable

        # Risk targets
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

def train_optimized_model(processed_filepath, batch_size=128, max_epochs=20):
    """
    Fast training function with optimizations.
    """
    print("Loading dataset with optimizations...")
    df = pd.read_parquet(processed_filepath)

    # Patient-based split
    unique_patients = df['PtID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)

    train_patient_ids = unique_patients[:int(len(unique_patients) * 0.8)]
    val_patient_ids = unique_patients[int(len(unique_patients) * 0.8):]

    train_df = df[df['PtID'].isin(train_patient_ids)]
    val_df = df[df['PtID'].isin(val_patient_ids)]

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")

    # Create optimized datasets
    train_dataset = FastCGMDataset(train_df, sequence_length=72)  # 6 hours
    val_dataset = FastCGMDataset(val_df, sequence_length=72)

    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2        # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )

    # Optimized model
    model = OptimizedGlucosePredictor(
        input_features=len(train_dataset.feature_cols),
        d_model=128,      # Smaller model
        n_heads=8,
        n_layers=4,       # Fewer layers
        n_patients=len(df['PtID'].unique()),
        learning_rate=2e-4
    )

    # Fast training setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,  # No accumulation for speed
        val_check_interval=0.5,     # Validate twice per epoch
        log_every_n_steps=50,       # Log less frequently
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
 
model, trainer = train_optimized_model('/content/drive/MyDrive/A glucose monitor/output/processed_glucose_data.parquet')
