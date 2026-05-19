"""
tests/test_churn_predict.py

不需要真实 AWS 凭证就能运行的单元测试。
用 unittest.mock 模拟 SageMaker 的响应，
确保工具的输入验证、输出结构、风险分级逻辑都正确。

运行:
    python -m pytest tests/test_churn_predict.py -v
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.churn_predict import churn_predict_tool, TOOL_SCHEMA, _assess_risk_factors


def make_mock_sagemaker(probability: float):
    """构造一个返回指定概率的 SageMaker mock"""
    mock_client = MagicMock()
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"predictions": [probability]}).encode()
    mock_client.invoke_endpoint.return_value = {"Body": mock_body}
    return mock_client


class TestChurnPredictTool(unittest.TestCase):

    @patch("tools.churn_predict.boto3.client")
    def test_high_risk_customer(self, mock_boto3):
        """高风险客户（月租客户 + 光纤 + 短期）应返回 high 级别"""
        mock_boto3.return_value = make_mock_sagemaker(0.85)

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

    @patch("tools.churn_predict.boto3.client")
    def test_low_risk_customer(self, mock_boto3):
        """低风险客户（两年合同 + 长期）应返回 low 级别"""
        mock_boto3.return_value = make_mock_sagemaker(0.12)

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

    @patch("tools.churn_predict.boto3.client")
    def test_medium_risk_customer(self, mock_boto3):
        mock_boto3.return_value = make_mock_sagemaker(0.55)

        result = churn_predict_tool(
            customer_id="C-003",
            tenure=12,
            contract="One year",
            internet_service="DSL",
            monthly_charges=60.0,
        )

        self.assertEqual(result["risk_level"], "medium")

    @patch("tools.churn_predict.boto3.client")
    def test_returns_required_keys(self, mock_boto3):
        """返回值必须包含 Agent 推理所需的所有字段"""
        mock_boto3.return_value = make_mock_sagemaker(0.5)

        result = churn_predict_tool(
            customer_id="C-004",
            tenure=6,
            contract="Month-to-month",
            internet_service="Fiber optic",
            monthly_charges=70.0,
        )

        required_keys = [
            "success", "customer_id", "churn_probability",
            "risk_level", "risk_score_pct", "key_risk_factors", "recommendation"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"缺少必填字段: {key}")

    @patch("tools.churn_predict.boto3.client")
    def test_sagemaker_failure_returns_error(self, mock_boto3):
        """SageMaker 调用失败时应返回 success=False，不抛出异常"""
        mock_boto3.return_value.invoke_endpoint.side_effect = Exception("EndpointNotFound")

        result = churn_predict_tool(
            customer_id="C-999",
            tenure=5,
            contract="Month-to-month",
            internet_service="DSL",
            monthly_charges=50.0,
        )

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("EndpointNotFound", result["error"])

    @patch("tools.churn_predict.boto3.client")
    def test_total_charges_defaults_to_monthly(self, mock_boto3):
        """未传入 total_charges 时应默认等于 monthly_charges"""
        mock_boto3.return_value = make_mock_sagemaker(0.3)

        # 检查 invoke_endpoint 被调用时 payload 里的 TotalCharges
        result = churn_predict_tool(
            customer_id="C-005",
            tenure=10,
            contract="One year",
            internet_service="DSL",
            monthly_charges=55.0,
            # 不传 total_charges
        )

        call_args = mock_boto3.return_value.invoke_endpoint.call_args
        payload = json.loads(call_args[1]["Body"])
        self.assertEqual(
            payload["instances"][0]["TotalCharges"],
            55.0,
            "未传 total_charges 时应默认等于 monthly_charges"
        )


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
