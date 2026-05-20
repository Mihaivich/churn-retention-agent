"""
tools/churn_predict.py  (更新版 — 支持本地 + SageMaker 双后端)

环境变量控制后端:
    CHURN_BACKEND=local      使用本地 FastAPI 服务 (默认，免费)
    CHURN_BACKEND=sagemaker  使用 AWS SageMaker (需要有效账号)
    LOCAL_ENDPOINT=http://localhost:8000

Agent 代码完全不需要改动，切换后端只需改一个环境变量。
"""

import json
import os
import requests
from dataclasses import dataclass
from typing import Optional

BACKEND = os.getenv("CHURN_BACKEND", "local")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "http://localhost:8000")
SAGEMAKER_ENDPOINT_NAME = "churn-predictor-endpoint"
REGION = "us-east-1"


@dataclass
class ChurnResult:
    customer_id: str
    churn_probability: float
    risk_level: str
    risk_score_pct: str
    key_risk_factors: list[str]
    recommendation: str


def _assess_risk_factors(features: dict) -> list[str]:
    factors = []
    if features.get("Contract") == "Month-to-month":
        factors.append("短期合同 (Month-to-month)，锁定性弱")
    if features.get("tenure", 99) < 12:
        factors.append(f"客户在网仅 {features.get('tenure')} 个月，尚未建立黏性")
    if features.get("InternetService") == "Fiber optic":
        factors.append("光纤用户历史流失率偏高")
    if features.get("PaymentMethod") == "Electronic check":
        factors.append("电子支票付款方式与流失强相关")
    if features.get("MonthlyCharges", 0) > 80:
        factors.append(f"月费 ${features.get('MonthlyCharges'):.1f}，高于平均水平")
    return factors or ["无明显高风险特征"]


def _build_payload(features: dict) -> dict:
    defaults = {
        "customerID": "agent-query",
        "gender": "Female", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0, "TotalCharges": 70.0,
    }
    return {"instances": [{**defaults, **features}]}


def _call_local(payload: dict) -> float:
    resp = requests.post(f"{LOCAL_ENDPOINT}/invocations", json=payload, timeout=2)
    resp.raise_for_status()
    return float(resp.json()["predictions"][0])


def _call_sagemaker(payload: dict) -> float:
    import boto3
    client = boto3.client("sagemaker-runtime", region_name=REGION)
    response = client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return float(json.loads(response["Body"].read())["predictions"][0])


def churn_predict_tool(
    customer_id: str,
    tenure: int,
    contract: str,
    internet_service: str,
    monthly_charges: float,
    total_charges: Optional[float] = None,
    payment_method: str = "Electronic check",
    **extra_features,
) -> dict:
    """
    预测客户流失概率。自动选择本地或 SageMaker 后端。
    通过环境变量 CHURN_BACKEND=local|sagemaker 切换。
    """
    if total_charges is None:
        total_charges = monthly_charges

    features = {
        "customerID": customer_id,
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "PaymentMethod": payment_method,
        **extra_features,
    }
    payload = _build_payload(features)

    try:
        prob = _call_sagemaker(payload) if BACKEND == "sagemaker" else _call_local(payload)
    except Exception as e:
        return {"success": False, "error": str(e), "customer_id": customer_id}

    if prob >= 0.7:
        risk_level, recommendation = "high", "立即介入：主动联系客户，提供个性化挽留方案"
    elif prob >= 0.4:
        risk_level, recommendation = "medium", "近期跟进：发送满意度调查，提前识别不满点"
    else:
        risk_level, recommendation = "low", "正常维护：可考虑升级销售或增值服务推荐"

    return {
        "success": True,
        "customer_id": customer_id,
        "churn_probability": prob,
        "risk_level": risk_level,
        "risk_score_pct": f"{prob:.0%}",
        "key_risk_factors": _assess_risk_factors(features),
        "recommendation": recommendation,
        "backend_used": BACKEND,
    }


TOOL_SCHEMA = {
    "name": "churn_predict",
    "description": "预测指定客户的流失概率，返回风险等级和行动建议。制定挽留方案前应首先调用此工具。",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户唯一标识符"},
            "tenure": {"type": "integer", "description": "客户在网月数，范围 0-72"},
            "contract": {
                "type": "string",
                "enum": ["Month-to-month", "One year", "Two year"],
                "description": "合同类型",
            },
            "internet_service": {
                "type": "string",
                "enum": ["DSL", "Fiber optic", "No"],
                "description": "互联网服务类型",
            },
            "monthly_charges": {"type": "number", "description": "客户每月费用（美元）"},
            "total_charges": {"type": "number", "description": "客户累计费用（可选）"},
            "payment_method": {
                "type": "string",
                "enum": [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)",
                ],
                "description": "付款方式",
            },
        },
        "required": ["customer_id", "tenure", "contract", "internet_service", "monthly_charges"],
    },
}
