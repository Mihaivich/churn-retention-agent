"""
tests/test_churn_predict.py

不需要真实 AWS 凭证或本地服务就能运行的单元测试。

现在 churn_predict.py 默认走本地 FastAPI（CHURN_BACKEND=local），
所以 mock 目标是 requests.post，而不是 boto3。
测试覆盖：输入验证、输出结构、风险分级、错误处理。

运行:
    python -m pytest tests/test_churn_predict.py -v
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用 local 后端（不依赖环境变量的当前值）
os.environ["CHURN_BACKEND"] = "local"

from tools.churn_predict import churn_predict_tool, TOOL_SCHEMA, _assess_risk_factors


def make_mock_response(probability: float) -> MagicMock:
    """构造一个模拟 requests.post 返回的 Response 对象"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"predictions": [probability]}
    return mock_resp


class TestChurnPredictTool(unittest.TestCase):

    @patch("tools.churn_predict.requests.post")
    def test_high_risk_customer(self, mock_post):
        """高风险客户（月租 + 光纤 + 短期）应返回 high 级别"""
        mock_post.return_value = make_mock_response(0.85)

        result = churn_predict_tool(
            customer_id="C-001",
            tenure=2,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=95.0,
            payment_method="Electronic check",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["churn_probability"], 0.7)
        self.assertIn("立即介入", result["recommendation"])
        self.assertEqual(result["risk_score_pct"], "85%")

    @patch("tools.churn_predict.requests.post")
    def test_low_risk_customer(self, mock_post):
        """低风险客户（两年合同 + 长期）应返回 low 级别"""
        mock_post.return_value = make_mock_response(0.12)

        result = churn_predict_tool(
            customer_id="C-002",
            tenure=48,
            contract="Two year",
            internet_service="DSL",
            monthly_charges=45.0,
            payment_method="Credit card (automatic)",
        )

        self.assertEqual(result["risk_level"], "low")
        self.assertIn("正常维护", result["recommendation"])

    @patch("tools.churn_predict.requests.post")
    def test_medium_risk_customer(self, mock_post):
        mock_post.return_value = make_mock_response(0.55)

        result = churn_predict_tool(
            customer_id="C-003",
            tenure=12,
            contract="One year",
            internet_service="DSL",
            monthly_charges=60.0,
        )

        self.assertEqual(result["risk_level"], "medium")

    @patch("tools.churn_predict.requests.post")
    def test_returns_required_keys(self, mock_post):
        """返回值必须包含 Agent 推理所需的所有字段"""
        mock_post.return_value = make_mock_response(0.5)

        result = churn_predict_tool(
            customer_id="C-004",
            tenure=6,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=70.0,
        )

        required_keys = [
            "success", "customer_id", "churn_probability",
            "risk_level", "risk_score_pct", "key_risk_factors",
            "recommendation", "backend_used",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"缺少必填字段: {key}")

    @patch("tools.churn_predict.requests.post")
    def test_local_failure_returns_error(self, mock_post):
        """本地服务调用失败时应返回 success=False，不抛出异常"""
        mock_post.side_effect = Exception("Connection refused")

        result = churn_predict_tool(
            customer_id="C-999",
            tenure=5,
            contract="Month-to-month",
            internet_service="DSL",
            monthly_charges=50.0,
        )

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Connection refused", result["error"])

    @patch("tools.churn_predict.requests.post")
    def test_total_charges_defaults_to_monthly(self, mock_post):
        """未传入 total_charges 时应默认等于 monthly_charges"""
        mock_post.return_value = make_mock_response(0.3)

        churn_predict_tool(
            customer_id="C-005",
            tenure=10,
            contract="One year",
            internet_service="DSL",
            monthly_charges=55.0,
            # 不传 total_charges
        )

        call_kwargs = mock_post.call_args[1]  # requests.post(..., json=payload)
        payload = call_kwargs["json"]
        self.assertEqual(
            payload["instances"][0]["TotalCharges"],
            55.0,
            "未传 total_charges 时应默认等于 monthly_charges",
        )

    @patch("tools.churn_predict.requests.post")
    def test_backend_used_field(self, mock_post):
        """返回值应标注使用了哪个后端"""
        mock_post.return_value = make_mock_response(0.4)

        result = churn_predict_tool(
            customer_id="C-006",
            tenure=8,
            contract="Month-to-month",
            internet_service="DSL",
            monthly_charges=60.0,
        )

        self.assertEqual(result["backend_used"], "local")


class TestRiskFactors(unittest.TestCase):

    def test_month_to_month_flagged(self):
        factors = _assess_risk_factors({"Contract": "Month-to-month"})
        self.assertTrue(any("短期合同" in f for f in factors))

    def test_new_customer_flagged(self):
        factors = _assess_risk_factors({"tenure": 3})
        self.assertTrue(any("3 个月" in f for f in factors))

    def test_stable_customer_no_flags(self):
        factors = _assess_risk_factors({
            "Contract": "Two year",
            "tenure": 36,
            "InternetService": "DSL",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 40.0,
        })
        self.assertIn("无明显高风险特征", factors)

    def test_high_monthly_charges_flagged(self):
        factors = _assess_risk_factors({"MonthlyCharges": 95.0})
        self.assertTrue(any("月费" in f for f in factors))


class TestToolSchema(unittest.TestCase):

    def test_schema_has_required_fields(self):
        """schema 必须符合 Anthropic tool_use API 格式"""
        self.assertIn("name", TOOL_SCHEMA)
        self.assertIn("description", TOOL_SCHEMA)
        self.assertIn("input_schema", TOOL_SCHEMA)
        schema = TOOL_SCHEMA["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("required", schema)

    def test_required_params_in_properties(self):
        props = TOOL_SCHEMA["input_schema"]["properties"]
        for req in TOOL_SCHEMA["input_schema"]["required"]:
            self.assertIn(req, props, f"必填参数 {req} 不在 properties 中")

    def test_schema_name(self):
        self.assertEqual(TOOL_SCHEMA["name"], "churn_predict")


if __name__ == "__main__":
    unittest.main(verbosity=2)
