from __future__ import annotations
import os, io, json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---- Paths (kök dizine göre güvenli) ----
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.joblib"))
CARD_PATH  = os.getenv("MODEL_CARD", str(BASE_DIR / "models" / "model_card.json"))
FEEDBACK_CSV = os.getenv("FEEDBACK_CSV", str(BASE_DIR / "data" / "user_feedback.csv"))

app = FastAPI(title="Ayzonova Exoplanet AI API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Model card + Model yükü ----
try:
    with open(CARD_PATH, "r", encoding="utf-8") as f:
        MODEL_CARD = json.load(f)
    FEATURES: List[str] = MODEL_CARD["features"]
    CLASSES: List[str]  = MODEL_CARD.get("classes", [])
    TARGET_COL: str     = MODEL_CARD.get("target", "label")
except Exception as e:
    raise RuntimeError(f"model_card.json okunamadı: {e}")

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"model.joblib yüklenemedi: {e}")

def coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    # Zorunlu sütun kontrolü
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "Eksik sütun(lar)", "missing": missing})
    X = df[FEATURES].copy()
    # sayısala çevir + sonsuzları/NaN'ları toparla
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    return X

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True, "n_features": len(FEATURES), "version": "1.1"}

@app.get("/features")
def get_features():
    return {"features": FEATURES, "classes": CLASSES, "target": TARGET_COL}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # CSV veya JSON kabul
    try:
        content = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".json"):
            arr = json.loads(content.decode("utf-8"))
            df = pd.DataFrame(arr if isinstance(arr, list) else [arr])
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {e}")

    X = coerce_df(df)
    try:
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            classes = list(getattr(MODEL, "classes_", CLASSES or []))
        else:
            # predict_proba yoksa tek-sıcak olasılık üret
            yhat = MODEL.predict(X)
            classes = CLASSES or list(pd.Series(yhat).unique())
            proba = np.zeros((len(X), len(classes)))
            for i, lab in enumerate(yhat):
                proba[i, classes.index(lab) if lab in classes else 0] = 1.0

        yhat_idx = np.argmax(proba, axis=1)
        results = []
        for i in range(len(X)):
            probs = {str(classes[j]): float(proba[i, j]) for j in range(len(classes))}
            pred = str(classes[yhat_idx[i]])
            results.append({"prediction": pred, "proba": probs})
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin sırasında hata: {e}")

@app.get("/explain")
def explain(topk: int = 15):
    imp = getattr(MODEL, "feature_importances_", None)
    if imp is None:
        return {"ok": True, "message": "Bu model feature_importances_ sağlamıyor."}
    order = np.argsort(imp)[::-1][:topk]
    items = [{"feature": FEATURES[i], "importance": float(imp[i])} for i in order]
    return {"ok": True, "importances": items}

# ---- Kullanıcı geri bildirimi: veri toplama ----
@app.post("/feedback")
async def feedback(
    file: UploadFile = File(None),
    json_rows: Optional[list] = Body(default=None),
    label_col: str = Query(default=None, description="Etiket sütunu adı (varsayılan: model_card.target)")
):
    label_col = label_col or TARGET_COL
    if file is None and json_rows is None:
        raise HTTPException(status_code=400, detail="CSV veya JSON veri gönderin.")

    try:
        if file is not None:
            content = await file.read()
            name = (file.filename or "").lower()
            if name.endswith(".json"):
                rows = json.loads(content.decode("utf-8"))
                df = pd.DataFrame(rows if isinstance(rows, list) else [rows])
            else:
                df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.DataFrame(json_rows if isinstance(json_rows, list) else [json_rows])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Veri okunamadı: {e}")

    miss_feat = [c for c in FEATURES if c not in df.columns]
    if miss_feat:
        raise HTTPException(status_code=400, detail={"error": "Eksik özellik(ler)", "missing": miss_feat})
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Etiket sütunu '{label_col}' bulunamadı.")

    Path(FEEDBACK_CSV).parent.mkdir(parents=True, exist_ok=True)
    header = not Path(FEEDBACK_CSV).exists()
    df.to_csv(FEEDBACK_CSV, mode="a", index=False, header=header)
    return {"ok": True, "stored_rows": int(df.shape[0])}

# ---- Basit yeniden eğitim (demo) ----
@app.post("/retrain")
def retrain(min_rows: int = 40, test_size: float = 0.2, random_state: int = 42):
    p = Path(FEEDBACK_CSV)
    if not p.exists():
        raise HTTPException(status_code=400, detail="Hiç geri bildirim verisi yok (data/user_feedback.csv yok).")

    df = pd.read_csv(p)
    need_cols = FEATURES + [TARGET_COL]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise HTTPException(status_code=400, detail={"error": "Eksik sütunlar feedback datasında", "missing": miss})
    if len(df) < min_rows:
        raise HTTPException(status_code=400, detail=f"Yetersiz veri: {len(df)} satır var, en az {min_rows} gerekli.")

    X = coerce_df(df)
    y = df[TARGET_COL].astype(str).str.strip()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = GradientBoostingClassifier(random_state=random_state)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    f1  = f1_score(yte, clf.predict(Xte), average="macro")

    # modeli güncelle
    try:
        old_path = MODEL_PATH + ".bak"
        if Path(MODEL_PATH).exists():
            Path(MODEL_PATH).replace(old_path)
        joblib.dump(clf, MODEL_PATH)
        global MODEL
        MODEL = clf
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model kaydedilemedi: {e}")

    return {"ok": True, "test_accuracy": acc, "test_f1_macro": f1, "rows_used": int(len(df))}
