"""
agent/runner.py

用 Anthropic tool_use API 把 churn_predict 工具接入 LLM。
这是最小可运行的 ReAct Agent 实现，不依赖 LangChain，
直接展示 Thought → Tool Call → Observation → Answer 循环。

运行前设置环境变量:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export AWS_DEFAULT_REGION="us-east-1"

运行:
    python agent/runner.py
"""

import json
import os
import anthropic

# 把工具函数和 schema 一起导入
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.churn_predict import churn_predict_tool, TOOL_SCHEMA


# ── 工具分发表 ────────────────────────────────────────────────────
# 后续添加 retrieve_cases、get_customer 等工具时，只需在这里注册
TOOLS = {
    "churn_predict": churn_predict_tool,
}

ALL_TOOL_SCHEMAS = [TOOL_SCHEMA]
# ─────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """你是一名电信公司的客户留存专家 Agent。

你的工作流程：
1. 收到客户查询后，先调用 churn_predict 工具评估流失风险
2. 根据风险等级和风险因素，制定具体的挽留方案
3. 给出清晰的行动建议，包括优先级和沟通话术

原则：
- 高风险客户（≥70%）需要立即介入，给出具体的优惠方案
- 中风险客户（40-70%）重点做满意度回访
- 低风险客户（<40%）可以考虑向上销售
- 每个建议都必须基于工具返回的具体风险因素，不要泛泛而谈
"""


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    运行一轮完整的 Agent 对话。

    实现了标准的 ReAct 循环：
    - LLM 思考 → 决定调用工具 → 执行工具 → 把结果反馈给 LLM → 最终回答

    Args:
        user_message: 用户的自然语言查询
        verbose:      是否打印每一步的推理过程（调试时很有用）

    Returns:
        Agent 的最终文字回答
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"用户: {user_message}")
        print(f"{'='*60}")

    # ReAct 循环：最多执行 5 轮工具调用，防止无限循环
    for step in range(5):

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOL_SCHEMAS,
            messages=messages,
        )

        # ── 情况 1：LLM 决定调用工具 ─────────────────────────────
        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if verbose:
                for tu in tool_uses:
                    print(f"\n[Step {step+1}] Agent 调用工具: {tu.name}")
                    print(f"  参数: {json.dumps(tu.input, ensure_ascii=False, indent=2)}")

            # 把 LLM 的回复（含工具调用请求）加入历史
            messages.append({"role": "assistant", "content": response.content})

            # 执行所有工具调用，收集结果
            tool_results = []
            for tu in tool_uses:
                if tu.name in TOOLS:
                    result = TOOLS[tu.name](**tu.input)
                else:
                    result = {"error": f"未知工具: {tu.name}"}

                if verbose:
                    print(f"\n[Step {step+1}] 工具返回:")
                    print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # 把工具结果反馈给 LLM，进入下一轮推理
            messages.append({"role": "user", "content": tool_results})

        # ── 情况 2：LLM 给出最终回答 ─────────────────────────────
        elif response.stop_reason == "end_turn":
            final_answer = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            if verbose:
                print(f"\n[最终回答]")
                print(f"{'-'*40}")
                print(final_answer)
            return final_answer

        else:
            # 意外的停止原因
            break

    return "Agent 未能完成推理，请检查工具配置。"


# ── 本地测试入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    # 模拟一个真实的客服查询场景
    test_query = """
    帮我评估这位客户的流失风险并给出挽留建议：
    客户ID: C-10023
    在网时长: 3 个月
    合同类型: Month-to-month
    网络服务: Fiber optic
    月费: $95.50
    累计费用: $286.50
    付款方式: Electronic check
    """

    run_agent(test_query, verbose=True)
