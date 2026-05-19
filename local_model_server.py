"""
local_model_server.py

用 FastAPI 在本地模拟 SageMaker endpoint，加载已有的 model.joblib。
接口格式与原 SageMaker endpoint 完全一致。

依赖: pip install fastapi uvicorn joblib scikit-learn pandas

运行: python local_model_server.py
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Churn Predictor (Local)", version="1.0")

# ── 加载模型 ──────────────────────────────────────────────────────
MODEL_PATH = Path("model/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"找不到模型文件: {MODEL_PATH}\n请先运行: dvc repro"
    )

model = joblib.load(MODEL_PATH)
print(f"模型加载成功: {MODEL_PATH}")

# 从模型本身读取期望的特征列，不再硬编码
FEATURE_COLUMNS = list(model.feature_names_in_)
print(f"特征列 ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")


# ── 请求/响应格式 ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    instances: list[dict]

class PredictResponse(BaseModel):
    predictions: list[float]


# ── 特征预处理 ────────────────────────────────────────────────────
DEFAULTS = {
    "customerID": "unknown",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0,
}

def preprocess(instance: dict) -> pd.DataFrame:
    """填充缺失字段，保持模型期望的列顺序"""
    row = {**DEFAULTS, **instance}           # 传入值覆盖默认值
    df = pd.DataFrame([row])
    # 补齐缺失列，按模型训练时的顺序排列
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, None)
    return df[FEATURE_COLUMNS]


# ── 预测接口 ──────────────────────────────────────────────────────
@app.post("/invocations", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        predictions = []
        for instance in request.instances:
            df = preprocess(instance)
            prob = float(model.predict_proba(df)[0][1])
            predictions.append(prob)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
async def health_check():
    return {"status": "healthy", "model": str(MODEL_PATH)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
