"""
local_model_server.py

用 FastAPI 在本地模拟 SageMaker endpoint，加载已有的 model.joblib。
接口格式与原 SageMaker endpoint 完全一致，Agent 代码无需修改。

依赖安装:
    pip install fastapi uvicorn joblib scikit-learn pandas

运行:
    python local_model_server.py
    # 服务启动在 http://localhost:8000

测试:
    curl -X POST http://localhost:8000/invocations \
      -H "Content-Type: application/json" \
      -d '{"instances": [{"tenure": 3, "Contract": "Month-to-month", ...}]}'
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Churn Predictor (Local)", version="1.0")

# ── 加载模型 ──────────────────────────────────────────────────────
# 你的 repo 里模型存在 model/model.joblib（DVC 管理）
# 先 dvc pull 把它拉到本地
MODEL_PATH = Path("model/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"找不到模型文件: {MODEL_PATH}\n"
        "请先运行: dvc pull\n"
        "或者: dvc repro (重新训练)"
    )

model = joblib.load(MODEL_PATH)
print(f"模型加载成功: {MODEL_PATH}")


# ── 请求/响应格式：与 SageMaker 保持一致 ─────────────────────────
class PredictRequest(BaseModel):
    instances: list[dict]


class PredictResponse(BaseModel):
    predictions: list[float]


# ── 特征预处理（与训练时保持一致）────────────────────────────────
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def preprocess(instance: dict) -> pd.DataFrame:
    """把原始 dict 转成模型期望的 DataFrame，去掉 customerID"""
    cleaned = {k: v for k, v in instance.items() if k != "customerID"}
    df = pd.DataFrame([cleaned])
    # 只保留训练时用到的列，缺失列填充众数默认值
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FEATURE_COLUMNS]


# ── 预测接口 ──────────────────────────────────────────────────────
@app.post("/invocations", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        predictions = []
        for instance in request.instances:
            df = preprocess(instance)
            # predict_proba 返回 [[prob_no_churn, prob_churn]]，取第二列
            prob = float(model.predict_proba(df)[0][1])
            predictions.append(prob)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ping")
async def health_check():
    """SageMaker 健康检查接口，保持兼容"""
    return {"status": "healthy", "model": str(MODEL_PATH)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
