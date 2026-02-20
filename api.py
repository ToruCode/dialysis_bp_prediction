# -*- coding: utf-8 -*-
"""
api.py — 透析中の血圧予測 API（FastAPI）
起動例:
  uvicorn api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import io, json, os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# ===================== 設定 =====================
SCALER_X_PATH   = Path("scaler_x.pkl")
SCALER_Y_PATH   = Path("scaler_y.pkl")
SCALER_T_PATH   = Path("scaler_t.pkl")
MODEL_PATH      = Path("model.pth")

FEATURE_NAMES = [
    "DW", "CTR", "50%TZ20ml", "リズミック", "レニベース",
    "積算除水量", "総除水量", "残し量", "下肢アップ", "液温"
]
TARGET_NAMES = ["収縮期血圧", "拡張期血圧", "脈拍"]
OUT_DIM = len(TARGET_NAMES)

# ===================== ユーティリティ =====================
class FallbackMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=[128, 64], out_dim=OUT_DIM, dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def load_scaler(path: Path):
    if not path.exists(): return None
    return joblib.load(path)

def load_model(model_path: Path, in_dim: int):
    device = torch.device("cpu")
    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval(); return m
    except Exception:
        pass
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, nn.Module):
        obj.eval(); return obj
    if isinstance(obj, dict):
        sd = None
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]; break
        if sd is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        model = FallbackMLP(in_dim, out_dim=OUT_DIM)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    raise RuntimeError("未知のモデル保存形式です。")

@torch.no_grad()
def predict_core(model: nn.Module, X_scaled: np.ndarray, scaler_y=None) -> np.ndarray:
    x = torch.from_numpy(X_scaled.astype(np.float32))
    y = model(x).cpu().numpy()
    if y.ndim == 1 and y.size == OUT_DIM:
        y = y.reshape(1, OUT_DIM)
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    if scaler_y is not None:
        try:
            y = scaler_y.inverse_transform(y)
        except Exception:
            pass
    return y

# ===================== FastAPI 準備 =====================
app = FastAPI(title="BP Predictor API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not SCALER_X_PATH.exists():
    raise RuntimeError(f"入力スケーラーが見つかりません: {SCALER_X_PATH}")
scaler_x = load_scaler(SCALER_X_PATH)
scaler_y = load_scaler(SCALER_Y_PATH) or load_scaler(SCALER_T_PATH)

if not MODEL_PATH.exists():
    raise RuntimeError(f"モデルが見つかりません: {MODEL_PATH}")
model = load_model(MODEL_PATH, in_dim=len(FEATURE_NAMES))

# ===================== スキーマ =====================
class PredictRequest(BaseModel):
    instances: List[Union[Dict[str, float], List[float]]]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, int]]  # ← int に変更
    n_features: int
    feature_names: List[str]
    target_names: List[str]
    mode: str

# ===================== ルーティング =====================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/meta")
def meta():
    return {
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "mode": "single_model",
    }

def build_X(instances):
    use_dict = any(isinstance(s, dict) for s in instances)
    if use_dict:
        df = pd.DataFrame(instances)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing: raise HTTPException(400, f"不足している列: {missing}")
        return df[FEATURE_NAMES].astype(float).values
    else:
        rows = []
        for row in instances:
            rows.append([float(v) for v in row])
        return np.array(rows, dtype=np.float32)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = build_X(req.instances)
    Xs = scaler_x.transform(X)
    Y = predict_core(model, Xs, scaler_y=scaler_y)
    if Y.shape[1] != OUT_DIM:
        raise HTTPException(400, f"想定外の出力次元: {Y.shape[1]} (期待={OUT_DIM})")
    outs = [
        {TARGET_NAMES[0]: int(round(y[0])),
         TARGET_NAMES[1]: int(round(y[1])),
         TARGET_NAMES[2]: int(round(y[2]))}
        for y in Y
    ]
    return PredictResponse(
        predictions=outs,
        n_features=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        mode="single_model",
    )

@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), encoding="cp932")
    if not set(FEATURE_NAMES).issubset(df.columns):
        miss = [c for c in FEATURE_NAMES if c not in df.columns]
        raise HTTPException(400, f"CSV に必要列がありません: {miss}")
    df_in = df[FEATURE_NAMES].copy().dropna()
    if len(df_in) == 0:
        raise HTTPException(400, "有効な行がありません")
    Xs = scaler_x.transform(df_in.values.astype(np.float32))
    Y = predict_core(model, Xs, scaler_y=scaler_y)
    if Y.shape[1] != OUT_DIM:
        raise HTTPException(400, f"想定外の出力次元: {Y.shape[1]} (期待={OUT_DIM})")
    outs = [
        {TARGET_NAMES[0]: int(round(y[0])),
         TARGET_NAMES[1]: int(round(y[1])),
         TARGET_NAMES[2]: int(round(y[2]))}
        for y in Y
    ]
    return PredictResponse(
        predictions=outs,
        n_features=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        mode="single_model",
    )
