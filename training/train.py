import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "Dataset"
CSV_FILE = os.path.join(DATASET_DIR, "train.csv")


# LOAD CSV


df = pd.read_csv(CSV_FILE)
df = df[df["target_name"] == "Dry_Total_g"].copy()

df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
df["Month"] = df["Sampling_Date"].dt.month
df["DayOfYear"] = df["Sampling_Date"].dt.dayofyear


# FEATURE ENGINEERING


df["NDVI_Height"] = df["Pre_GSHH_NDVI"] * df["Height_Ave_cm"]
df["Height_Sq"] = df["Height_Ave_cm"] ** 2
df["NDVI_Sq"] = df["Pre_GSHH_NDVI"] ** 2

le_species = LabelEncoder()
le_state = LabelEncoder()

df["Species"] = le_species.fit_transform(df["Species"])
df["State"] = le_state.fit_transform(df["State"])

tabular_cols = [
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
    "NDVI_Height",
    "Height_Sq",
    "NDVI_Sq",
    "Species",
    "State",
    "Month",
    "DayOfYear"
]


# IMAGE DATASET FOR SSL


class SSLDataset(Dataset):

    def __init__(self, img_paths, transform):
        self.paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = Image.open(self.paths[idx]).convert("RGB")

        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2


ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.ToTensor()
])

image_paths = []

for p in df["image_path"].unique():
    full = os.path.join(DATASET_DIR, p)
    if os.path.exists(full):
        image_paths.append(full)

ssl_dataset = SSLDataset(image_paths, ssl_transform)
ssl_loader = DataLoader(ssl_dataset, batch_size=16, shuffle=True)


# SELF SUPERVISED MODEL


encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
encoder.fc = nn.Identity()

projection = nn.Sequential(
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,128)
)

model_ssl = nn.Sequential(encoder, projection).to(DEVICE)

optimizer = torch.optim.Adam(model_ssl.parameters(), lr=1e-4)

print("Starting self-supervised training...")

for epoch in range(10):

    total_loss = 0

    for x1, x2 in ssl_loader:

        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)

        z1 = nn.functional.normalize(model_ssl(x1), dim=1)
        z2 = nn.functional.normalize(model_ssl(x2), dim=1)

        # NT-Xent style loss: maximize agreement between positive pairs
        # and minimize agreement with all other samples in the batch
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / 0.5   # temperature=0.5
        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=DEVICE).bool()
        sim.masked_fill_(mask, float('-inf'))
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=DEVICE),
            torch.arange(0, batch_size, device=DEVICE)
        ])
        loss = nn.functional.cross_entropy(sim, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("SSL Epoch:", epoch+1, "Loss:", total_loss)

torch.save(encoder.state_dict(), "ssl_encoder.pth")

print("Self-supervised encoder saved.")


# IMAGE FEATURE EXTRACTION


encoder.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_features(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)

    return feat.cpu().numpy().flatten()


print("Extracting image features...")

image_feats = []
tabular_feats = []
targets = []

for _, row in df.iterrows():

    path = os.path.join(DATASET_DIR, row["image_path"])

    if not os.path.exists(path):
        continue

    try:

        img_feat = extract_features(path)

        image_feats.append(img_feat)
        tabular_feats.append(row[tabular_cols].values.astype(np.float32))
        targets.append(row["target"])

    except:
        continue


image_feats = np.array(image_feats)
tabular_feats = np.array(tabular_feats)
targets = np.array(targets)

print("Original image feature shape:", image_feats.shape)


# PCA


pca = PCA(n_components=64)
image_feats = pca.fit_transform(image_feats)

print("After PCA:", image_feats.shape)


# FINAL FEATURES


X = np.concatenate([image_feats, tabular_feats], axis=1)
y = np.log1p(targets)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Final feature shape:", X.shape)


# K-FOLD TRAINING


kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_all = []
y_all = []

best_r2 = -np.inf
best_xgb = None
best_lgb = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print("\nTraining Fold", fold+1)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )

    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_samples=10
    )

    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)

    pred_xgb = xgb.predict(X_val)
    pred_lgb = lgb.predict(X_val)

    pred = (pred_xgb + pred_lgb) / 2

    fold_r2 = r2_score(y_val, pred)
    print(f"  Fold {fold+1} R²: {fold_r2:.4f}")

    if fold_r2 > best_r2:
        best_r2 = fold_r2
        best_xgb = xgb
        best_lgb = lgb

    preds_all.extend(pred)
    y_all.extend(y_val)


# METRICS


preds_all = np.array(preds_all)
y_all = np.array(y_all)

y_real = np.expm1(y_all)
pred_real = np.expm1(preds_all)

r2 = r2_score(y_real, pred_real)
rmse = np.sqrt(mean_squared_error(y_real, pred_real))
mae = mean_absolute_error(y_real, pred_real)

print("\nMODEL PERFORMANCE")
print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)


# SAVE


os.makedirs("saved_models", exist_ok=True)

joblib.dump(best_xgb, "saved_models/xgb_model.pkl")
joblib.dump(best_lgb, "saved_models/lgb_model.pkl")
joblib.dump(pca, "saved_models/pca.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(le_species, "saved_models/le_species.pkl")
joblib.dump(le_state, "saved_models/le_state.pkl")

print("Models saved.")
