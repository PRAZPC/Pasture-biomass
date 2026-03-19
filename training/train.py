"""
Biomass Estimation Training Script
==================================
Optimized for RTX 3060 Ti GPU with advanced SSL training and ensemble models.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


########################################
# GPU CONFIGURATION
########################################

def setup_gpu():
    """Configure GPU for optimal performance on RTX 3060 Ti."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Detected: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        
        # Enable TF32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected, using CPU")
    
    return device


DEVICE = setup_gpu()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "Dataset"
CSV_FILE = DATASET_DIR / "train.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


########################################
# LOAD DATA
########################################

print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

df = pd.read_csv(CSV_FILE)
print(f"Raw samples: {len(df)}")

# Pivot targets
df = df.pivot_table(
    index=[
        "sample_id",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm"
    ],
    columns="target_name",
    values="target"
).reset_index()

df = df.dropna(subset=["Dry_Total_g"])
df.fillna(0, inplace=True)
print(f"After pivot & cleaning: {len(df)} samples")


########################################
# ADVANCED DATA AUGMENTATION
########################################

class SSLDataset(Dataset):
    """Self-supervised learning dataset with advanced augmentations."""
    
    def __init__(self, paths, transform1, transform2):
        self.paths = paths
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x1 = self.transform1(img)
        x2 = self.transform2(img)
        return x1, x2


# Strong augmentations for SSL (SimCLR-style)
ssl_transform_strong = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ssl_transform_weak = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Collect image paths
image_paths = []
for p in df["image_path"].unique():
    full = os.path.join(DATASET_DIR, p)
    if os.path.exists(full):
        image_paths.append(full)

print(f"Found {len(image_paths)} unique images")


########################################
# SSL MODEL (SimCLR-style)
########################################

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.07):
    """NT-Xent loss (Normalized Temperature-scaled Cross Entropy)."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    
    sim = torch.mm(z, z.T) / temperature
    
    # Create mask for positive pairs
    mask = torch.eye(2 * batch_size, device=DEVICE).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Labels: positive pairs are at distance batch_size
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=DEVICE),
        torch.arange(0, batch_size, device=DEVICE)
    ])
    
    return nn.functional.cross_entropy(sim, labels)


########################################
# SSL TRAINING
########################################

print("\n" + "="*50)
print("SELF-SUPERVISED LEARNING")
print("="*50)

# Use ResNet-50 for better features (RTX 3060 Ti can handle it)
encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
encoder_dim = encoder.fc.in_features  # 2048 for ResNet-50
encoder.fc = nn.Identity()

projection = ProjectionHead(input_dim=encoder_dim, hidden_dim=512, output_dim=128)

encoder = encoder.to(DEVICE)
projection = projection.to(DEVICE)

# Dataset and DataLoader - larger batch size for RTX 3060 Ti (8GB VRAM)
ssl_dataset = SSLDataset(image_paths, ssl_transform_strong, ssl_transform_weak)
ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=32,  # Increased for better GPU utilization
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(projection.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

# Cosine annealing scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Training parameters
SSL_EPOCHS = 25
ssl_losses = []

print(f"Training for {SSL_EPOCHS} epochs...")
print(f"Batch size: 32, Batches per epoch: {len(ssl_loader)}")

for epoch in range(SSL_EPOCHS):
    encoder.train()
    projection.train()
    
    epoch_loss = 0
    for batch_idx, (x1, x2) in enumerate(ssl_loader):
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        
        # Forward pass
        h1 = encoder(x1)
        h2 = encoder(x2)
        z1 = projection(h1)
        z2 = projection(h2)
        
        loss = nt_xent_loss(z1, z2, temperature=0.07)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_loss = epoch_loss / len(ssl_loader)
    ssl_losses.append(avg_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d}/{SSL_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

# Save SSL encoder
torch.save(encoder.state_dict(), PROJECT_ROOT / "training" / "ssl_encoder.pth")
print("✓ SSL encoder saved")


# Plot SSL training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, SSL_EPOCHS + 1), ssl_losses, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('NT-Xent Loss', fontsize=12)
plt.title('Self-Supervised Learning Loss Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ssl_loss_curve.png", dpi=150)
plt.close()
print("✓ SSL loss plot saved")


########################################
# FEATURE EXTRACTION
########################################

print("\n" + "="*50)
print("FEATURE EXTRACTION")
print("="*50)

encoder.eval()

# Standard transform for inference
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features(path):
    """Extract features from a single image."""
    img = Image.open(path).convert("RGB")
    img = inference_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = encoder(img)
    return feat.cpu().numpy().flatten()


print("Extracting features from all images...")
image_feats = []
targets = []
valid_indices = []

for idx, row in df.iterrows():
    path = os.path.join(DATASET_DIR, row["image_path"])
    if not os.path.exists(path):
        continue
    try:
        img_feat = extract_features(path)
        image_feats.append(img_feat)
        targets.append(row["Dry_Total_g"])
        valid_indices.append(idx)
    except Exception as e:
        continue
    
    if len(image_feats) % 100 == 0:
        print(f"  Processed {len(image_feats)} images...")

image_feats = np.array(image_feats)
targets = np.array(targets)

print(f"✓ Extracted features from {len(image_feats)} images")
print(f"  Feature dimension: {image_feats.shape[1]}")


########################################
# PCA DIMENSIONALITY REDUCTION
########################################

print("\n" + "="*50)
print("PCA DIMENSIONALITY REDUCTION")
print("="*50)

# Use more PCA components since we have ResNet-50 features
n_components = 128
pca = PCA(n_components=n_components)
image_feats_pca = pca.fit_transform(image_feats)

explained_var = np.cumsum(pca.explained_variance_ratio_)
print(f"PCA: {n_components} components explain {explained_var[-1]*100:.1f}% variance")

# Plot PCA explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, label='Individual')
plt.plot(range(1, n_components + 1), explained_var, 'r-o', markersize=2, label='Cumulative')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('PCA Explained Variance', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_variance.png", dpi=150)
plt.close()
print("✓ PCA variance plot saved")


########################################
# PREPARE FINAL FEATURES
########################################

X = image_feats_pca
y = np.log1p(targets)  # Log transform for better distribution

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X = scaler.fit_transform(X)

print(f"Final feature shape: {X.shape}")
print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")


########################################
# K-FOLD TRAINING WITH ENSEMBLE
########################################

print("\n" + "="*50)
print("K-FOLD CROSS-VALIDATION TRAINING")
print("="*50)

# Check if GPU is available for XGBoost/LightGBM
USE_GPU_BOOST = torch.cuda.is_available()
if USE_GPU_BOOST:
    print("✓ Using GPU for XGBoost/LightGBM")
else:
    print("⚠ Using CPU for XGBoost/LightGBM")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for results
fold_scores = {'r2': [], 'rmse': [], 'mae': []}
all_preds = np.zeros_like(y)
all_actual = np.zeros_like(y)

best_r2 = -np.inf
best_xgb = None
best_lgb = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # XGBoost with optimized parameters
    xgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
    }
    if USE_GPU_BOOST:
        xgb_params['device'] = 'cuda'
    
    xgb = XGBRegressor(**xgb_params)
    
    # LightGBM with optimized parameters
    lgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'random_state': 42,
    }
    if USE_GPU_BOOST:
        lgb_params['device'] = 'gpu'
    
    lgb = LGBMRegressor(**lgb_params)
    
    # Train models
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Ensemble prediction
    pred_xgb = xgb.predict(X_val)
    pred_lgb = lgb.predict(X_val)
    pred = (pred_xgb + pred_lgb) / 2
    
    # Store predictions
    all_preds[val_idx] = pred
    all_actual[val_idx] = y_val
    
    # Calculate metrics (in original scale)
    y_val_real = np.expm1(y_val)
    pred_real = np.expm1(pred)
    
    fold_r2 = r2_score(y_val_real, pred_real)
    fold_rmse = np.sqrt(mean_squared_error(y_val_real, pred_real))
    fold_mae = mean_absolute_error(y_val_real, pred_real)
    
    fold_scores['r2'].append(fold_r2)
    fold_scores['rmse'].append(fold_rmse)
    fold_scores['mae'].append(fold_mae)
    
    print(f"  R²: {fold_r2:.4f} | RMSE: {fold_rmse:.2f} | MAE: {fold_mae:.2f}")
    
    if fold_r2 > best_r2:
        best_r2 = fold_r2
        best_xgb = xgb
        best_lgb = lgb


########################################
# FINAL METRICS & PLOTS
########################################

print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE")
print("="*50)

# Convert to original scale
y_real = np.expm1(all_actual)
pred_real = np.expm1(all_preds)

final_r2 = r2_score(y_real, pred_real)
final_rmse = np.sqrt(mean_squared_error(y_real, pred_real))
final_mae = mean_absolute_error(y_real, pred_real)

print(f"\n{'='*40}")
print(f"  Overall R²:   {final_r2:.4f}")
print(f"  Overall RMSE: {final_rmse:.2f}")
print(f"  Overall MAE:  {final_mae:.2f}")
print(f"{'='*40}")

print(f"\nCross-Validation Statistics:")
print(f"  R²:   {np.mean(fold_scores['r2']):.4f} ± {np.std(fold_scores['r2']):.4f}")
print(f"  RMSE: {np.mean(fold_scores['rmse']):.2f} ± {np.std(fold_scores['rmse']):.2f}")
print(f"  MAE:  {np.mean(fold_scores['mae']):.2f} ± {np.std(fold_scores['mae']):.2f}")


# Plot 1: Predictions vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.scatter(y_real, pred_real, alpha=0.5, s=20, c='steelblue', edgecolors='none')
max_val = max(y_real.max(), pred_real.max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Biomass (g)', fontsize=12)
ax1.set_ylabel('Predicted Biomass (g)', fontsize=12)
ax1.set_title(f'Predictions vs Actual\nR² = {final_r2:.4f}', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residual Distribution
ax2 = axes[1]
residuals = pred_real - y_real
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Residual Distribution\nMean: {residuals.mean():.2f}, Std: {residuals.std():.2f}', 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prediction_analysis.png", dpi=150)
plt.close()
print("✓ Prediction analysis plot saved")


# Plot 3: Cross-validation performance
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(5)
width = 0.25

bars1 = ax.bar(x - width, fold_scores['r2'], width, label='R² Score', color='steelblue')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Cross-Validation R² Scores by Fold', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.axhline(y=np.mean(fold_scores['r2']), color='red', linestyle='--', 
           label=f'Mean R² = {np.mean(fold_scores["r2"]):.4f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "cv_performance.png", dpi=150)
plt.close()
print("✓ CV performance plot saved")


# Plot 4: Feature Importance (combined from both models)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XGBoost feature importance
xgb_importance = best_xgb.feature_importances_
top_n = 20
top_indices_xgb = np.argsort(xgb_importance)[-top_n:]
axes[0].barh(range(top_n), xgb_importance[top_indices_xgb], color='steelblue')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([f'PCA_{i}' for i in top_indices_xgb])
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('XGBoost Feature Importance (Top 20)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# LightGBM feature importance
lgb_importance = best_lgb.feature_importances_
top_indices_lgb = np.argsort(lgb_importance)[-top_n:]
axes[1].barh(range(top_n), lgb_importance[top_indices_lgb], color='forestgreen')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels([f'PCA_{i}' for i in top_indices_lgb])
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('LightGBM Feature Importance (Top 20)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
plt.close()
print("✓ Feature importance plot saved")


# Plot 5: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(targets, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
axes[0].set_xlabel('Biomass (g)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Target Distribution (Original)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(y, bins=50, color='forestgreen', edgecolor='white', alpha=0.7)
axes[1].set_xlabel('log(1 + Biomass)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Target Distribution (Log-transformed)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=150)
plt.close()
print("✓ Target distribution plot saved")


########################################
# SAVE MODELS
########################################

print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

joblib.dump(best_xgb, SAVED_MODELS_DIR / "xgb_model.pkl")
joblib.dump(best_lgb, SAVED_MODELS_DIR / "lgb_model.pkl")
joblib.dump(pca, SAVED_MODELS_DIR / "pca.pkl")
joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

print(f"✓ Models saved to: {SAVED_MODELS_DIR}")
print(f"✓ Plots saved to: {PLOTS_DIR}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"\nGenerated plots:")
print(f"  - {PLOTS_DIR / 'ssl_loss_curve.png'}")
print(f"  - {PLOTS_DIR / 'pca_variance.png'}")
print(f"  - {PLOTS_DIR / 'prediction_analysis.png'}")
print(f"  - {PLOTS_DIR / 'cv_performance.png'}")
print(f"  - {PLOTS_DIR / 'feature_importance.png'}")
print(f"  - {PLOTS_DIR / 'target_distribution.png'}")
