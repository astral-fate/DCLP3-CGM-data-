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
    Specialized multi-head attention that focuses on glucose-specific patterns:
    - Short-term trends (15-30 min)
    - Medium-term patterns (1-2 hours)
    - Long-term circadian rhythms (daily)
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Multi-head projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
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
    """Single transformer block optimized for glucose prediction."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadGlucoseAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Better than ReLU for this application
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

class HierarchicalGlucosePredictor(pl.LightningModule):
    """
    Hierarchical Transformer for glucose prediction with 9M+ samples.

    Architecture designed for:
    - Multi-scale temporal attention (minutes to hours to days)
    - Patient-specific embeddings for personalization
    - Clinical-aware loss functions
    """

    def __init__(self,
                 input_features=10,
                 d_model=256,
                 n_heads=8,
                 n_layers=6,
                 d_ff=1024,
                 max_seq_length=288,  # 24 hours at 5-min intervals
                 n_patients=168,
                 prediction_horizon=6,  # 30 minutes ahead
                 dropout=0.1,
                 learning_rate=1e-4):
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

        # Multi-scale output heads
        self.glucose_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, prediction_horizon)  # Predict multiple steps
        )

        # Trend classification head (↗️↘️ arrows)
        self.trend_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 5)  # 5 trend categories
        )

        # Risk assessment head (hypo/hyper probability)
        self.risk_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2)  # P(hypo), P(hyper)
        )

        self.dropout = nn.Dropout(dropout)

        # Loss weights for clinical priorities
        self.glucose_weight = 1.0
        self.trend_weight = 0.3
        self.risk_weight = 0.5

    def forward(self, x, patient_ids, mask=None):
        batch_size, seq_len, _ = x.shape

        # Input embedding and positional encoding
        x = self.input_projection(x)  # [batch, seq, d_model]

        # Add patient-specific embeddings
        # FIXED: Remove the extra unsqueeze(1) since patient_ids already has correct shape after embedding
        patient_emb = self.patient_embedding(patient_ids)  # [batch, 1, d_model]
        patient_emb = patient_emb.expand(-1, seq_len, -1)  # [batch, seq, d_model]
        x = x + patient_emb

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq, batch, d_model] for positional encoding
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch, seq, d_model]

        x = self.dropout(x)

        # Pass through transformer layers
        attention_weights = []
        for transformer_layer in self.transformer_layers:
            x, attn_weights = transformer_layer(x, mask)
            attention_weights.append(attn_weights)

        # Use the last timestep for prediction (sequence-to-one)
        final_representation = x[:, -1, :]  # [batch, d_model]

        # Multi-task outputs
        glucose_pred = self.glucose_predictor(final_representation)  # [batch, horizon]
        trend_pred = self.trend_classifier(final_representation)     # [batch, 5]
        risk_pred = self.risk_predictor(final_representation)  # [batch, 2] - Remove sigmoid, use with BCEWithLogitsLoss

        return {
            'glucose': glucose_pred,
            'trend': trend_pred,
            'risk': risk_pred,
            'attention_weights': attention_weights[-1]  # Return last layer attention
        }

    def clinical_loss(self, predictions, targets):
        """
        Clinical-aware loss function that prioritizes dangerous ranges.
        """
        glucose_pred = predictions['glucose'][:, 0]  # First prediction (30 min ahead)
        glucose_target = targets['glucose'].squeeze()  # Remove extra dimension: [batch, 1] -> [batch]

        # Base MSE loss
        mse_loss = nn.MSELoss()(glucose_pred, glucose_target)

        # Clinical weighting: heavily penalize dangerous predictions
        error = torch.abs(glucose_pred - glucose_target)

        # Hypoglycemia zone (< 70): 3x weight
        hypo_mask = glucose_target < 70
        hypo_penalty = torch.mean(error[hypo_mask] * 3.0) if hypo_mask.sum() > 0 else 0

        # Severe hyperglycemia (> 250): 2x weight
        severe_hyper_mask = glucose_target > 250
        severe_hyper_penalty = torch.mean(error[severe_hyper_mask] * 2.0) if severe_hyper_mask.sum() > 0 else 0

        # Trend classification loss
        if 'trend' in targets:
            # Remove extra dimension and ensure target dtype is Long for CrossEntropyLoss
            trend_target = targets['trend'].squeeze().long()  # [batch, 1] -> [batch]
            trend_loss = nn.CrossEntropyLoss()(predictions['trend'], trend_target)
        else:
            trend_loss = 0

        # Risk prediction loss
        if 'risk' in targets:
            risk_loss = nn.BCEWithLogitsLoss()(predictions['risk'], targets['risk'])
        else:
            risk_loss = 0

        # Combined loss
        total_loss = (self.glucose_weight * (mse_loss + hypo_penalty + severe_hyper_penalty) +
                     self.trend_weight * trend_loss +
                     self.risk_weight * risk_loss)

        return total_loss, {
            'mse': mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss,
            'hypo_penalty': hypo_penalty.item() if isinstance(hypo_penalty, torch.Tensor) else hypo_penalty,
            'severe_hyper_penalty': severe_hyper_penalty.item() if isinstance(severe_hyper_penalty, torch.Tensor) else severe_hyper_penalty,
            'trend_loss': trend_loss.item() if isinstance(trend_loss, torch.Tensor) else trend_loss,
            'risk_loss': risk_loss.item() if isinstance(risk_loss, torch.Tensor) else risk_loss
        }

    def training_step(self, batch, batch_idx):
        predictions = self(batch['features'], batch['patient_ids'], batch.get('mask'))
        loss, loss_components = self.clinical_loss(predictions, batch['targets'])

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_mse', loss_components['mse'])
        self.log('train_hypo_penalty', loss_components['hypo_penalty'])

        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch['features'], batch['patient_ids'], batch.get('mask'))
        loss, loss_components = self.clinical_loss(predictions, batch['targets'])

        # Clinical metrics
        glucose_pred = predictions['glucose'][:, 0]
        glucose_target = batch['targets']['glucose'].squeeze()  # Remove extra dimension

        # MARD (Mean Absolute Relative Difference)
        # Avoid division by zero if target is 0 (though unlikely with glucose)
        mard = torch.mean(torch.abs(glucose_pred - glucose_target) / (glucose_target + 1e-6)) * 100

        self.log('val_loss', loss)
        self.log('val_mard', mard)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

class LargeScaleCGMDataset(Dataset):
    """
    Efficient dataset class for 9M+ glucose samples with sequence processing.
    """

    def __init__(self, df, sequence_length=144, prediction_horizon=6, patient_col='PtID'):
        self.df = df.sort_values([patient_col, 'DataDtTm']).reset_index(drop=True)
        self.sequence_length = sequence_length  # 12 hours at 5-min intervals
        self.prediction_horizon = prediction_horizon
        self.patient_col = patient_col

        # Create patient mapping
        unique_patients = df[patient_col].unique()
        self.patient_to_id = {patient: idx for idx, patient in enumerate(unique_patients)}

        # Pre-compute valid sequence indices for efficiency
        self.valid_indices = self._compute_valid_indices()

        # Feature columns (exclude metadata and targets)
        self.feature_cols = [col for col in df.columns
                           if col not in [patient_col, 'DataDtTm', 'glucose_target', 'Period', 'RecID', 'DataDtTm_adjusted', 'BolusType']]
        print(f"Dataset initialized with {len(self.feature_cols)} feature columns.")

    def _compute_valid_indices(self):
        """Pre-compute indices where we have valid sequences."""
        valid_indices = []

        for patient in self.df[self.patient_col].unique():
            patient_data = self.df[self.df[self.patient_col] == patient]

            # Only use indices where we have enough history + future
            # Ensure we have at least sequence_length points BEFORE the current index
            # and prediction_horizon points AFTER the current index
            for i in range(self.sequence_length,
                          len(patient_data) - self.prediction_horizon):
                # Get the global index from the original dataframe
                global_index = patient_data.index[i]
                valid_indices.append(global_index)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        central_idx = self.valid_indices[idx]

        # Get patient ID
        patient = self.df.loc[central_idx, self.patient_col]
        patient_id = self.patient_to_id[patient]

        # Extract sequence (past data)
        start_idx = central_idx - self.sequence_length
        end_idx = central_idx

        # Ensure sequence_data is numeric (float32) and fill NaNs
        sequence_data = self.df.loc[start_idx:end_idx-1, self.feature_cols].astype(np.float32).fillna(0).values

        # Extract target (future glucose)
        target_idx = central_idx + self.prediction_horizon
        glucose_target = self.df.loc[target_idx, 'glucose_target']

        # Create trend target based on glucose change
        current_glucose = self.df.loc[central_idx, 'CGM']
        glucose_change = glucose_target - current_glucose

        # Trend classification: 0=rapid_fall, 1=fall, 2=stable, 3=rise, 4=rapid_rise
        # Define thresholds based on clinical relevance (e.g., >3 mg/dL/min is rapid)
        # Assuming 5-min intervals, 3 mg/dL/min = 15 mg/dL / 5 min
        if glucose_change <= -15: # Rapid Fall
            trend_target = 0
        elif glucose_change <= -3: # Fall
            trend_target = 1
        elif abs(glucose_change) < 3: # Stable
            trend_target = 2
        elif glucose_change < 15: # Rise
            trend_target = 3
        else: # Rapid Rise
            trend_target = 4

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

# Training script for your 9M+ dataset
def train_large_scale_model(processed_filepath, batch_size=32, max_epochs=50):
    """
    Train the transformer model on your massive dataset.
    """
    print("Loading 9M+ sample dataset...")
    df = pd.read_parquet(processed_filepath)

    # Train/val split (time-based)
    # A better split for time series is by patient or time.
    # Let's split by patient IDs to avoid leakage.
    unique_patients = df['PtID'].unique()
    np.random.seed(42) # for reproducibility
    np.random.shuffle(unique_patients)

    # Split patients into train (80%) and validation (20%) sets
    train_patient_ids = unique_patients[:int(len(unique_patients) * 0.8)]
    val_patient_ids = unique_patients[int(len(unique_patients) * 0.8):]

    train_df = df[df['PtID'].isin(train_patient_ids)].reset_index(drop=True)
    val_df = df[df['PtID'].isin(val_patient_ids)].reset_index(drop=True)

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Number of train patients: {len(train_patient_ids)}")
    print(f"Number of val patients: {len(val_patient_ids)}")

    # Create datasets
    # Ensure the number of patients passed to the model is the total number of unique patients in the original df
    total_unique_patients = len(df['PtID'].unique())
    train_dataset = LargeScaleCGMDataset(train_df, sequence_length=144)
    val_dataset = LargeScaleCGMDataset(val_df, sequence_length=144)

    # Data loaders with optimal settings for large datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2, pin_memory=True)  # Reduced workers to 2
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)  # Reduced workers to 2

    # Initialize model
    model = HierarchicalGlucosePredictor(
        input_features=len(train_dataset.feature_cols),
        d_model=256,
        n_heads=8,
        n_layers=6,
        n_patients=total_unique_patients, # Use total number of patients for embedding layer size
        learning_rate=1e-4
    )

    # Training setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Fixed precision format
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = batch_size * 2
        val_check_interval=0.25,  # Validate 4 times per epoch
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_mard', mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_mard', patience=5),
            pl.callbacks.LearningRateMonitor()
        ]
    )

    print("Starting training on 9M+ samples...")
    trainer.fit(model, train_loader, val_loader)

    return model, trainer

# Usage for your dataset:
model, trainer = train_large_scale_model('/content/drive/MyDrive/A glucose monitor/output/processed_glucose_data.parquet')
