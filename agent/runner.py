"""
agent/runner.py  — Google Gemini 版本 (google-genai SDK)

运行前:
    pip install google-genai
    export GEMINI_API_KEY="AIza..."

运行:
    python agent/runner.py
"""

import json
import os
import sys
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.churn_predict import churn_predict_tool, TOOL_SCHEMA

# ── 配置客户端 ────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.1-flash-lite"

# ── 工具分发表 ────────────────────────────────────────────────────
TOOLS = {"churn_predict": churn_predict_tool}

# ── 把 Anthropic schema 转成 Gemini FunctionDeclaration ──────────
def _to_gemini_schema(prop: dict) -> dict:
    """单个参数的 JSON Schema → Gemini 格式"""
    result = {"type": prop.get("type", "string").upper()}
    if "description" in prop:
        result["description"] = prop["description"]
    if "enum" in prop:
        result["enum"] = prop["enum"]
    return result

def build_gemini_tool(schema: dict) -> types.Tool:
    props = {
        k: _to_gemini_schema(v)
        for k, v in schema["input_schema"]["properties"].items()
    }
    fn = types.FunctionDeclaration(
        name=schema["name"],
        description=schema["description"],
        parameters={
            "type": "OBJECT",
            "properties": props,
            "required": schema["input_schema"].get("required", []),
        },
    )
    return types.Tool(function_declarations=[fn])

GEMINI_TOOLS = [build_gemini_tool(TOOL_SCHEMA)]

SYSTEM_PROMPT = """你是一名电信公司的客户留存专家 Agent。

工作流程：
1. 先调用 churn_predict 工具评估客户流失风险
2. 根据风险等级和风险因素，制定具体的挽留方案
3. 给出清晰的行动建议，包括优先级和沟通话术

原则：
- 高风险（≥70%）：立即介入，给出具体优惠方案
- 中风险（40-70%）：近期跟进，做满意度回访
- 低风险（<40%）：正常维护，可考虑向上销售
- 每个建议必须基于工具返回的具体风险因素"""


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    ReAct 循环：send_message → function_call → send tool result → final answer
    google-genai SDK 用 contents 列表维护对话历史。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"用户: {user_message.strip()}")
        print(f"{'='*60}")

    contents = [types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )]

    for step in range(5):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
            ),
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)   # 把模型回复加入历史

        # 检查是否有 function call
        fn_calls = [p.function_call for p in candidate.content.parts
                    if p.function_call is not None]

        if not fn_calls:
            # 没有工具调用 → 最终回答
            final = "".join(
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            )
            if verbose:
                print(f"\n[最终回答]\n{'-'*40}\n{final}")
            return final

        # 执行工具，收集所有结果
        tool_result_parts = []
        for fn in fn_calls:
            args = dict(fn.args)
            if verbose:
                print(f"\n[Step {step+1}] 调用工具: {fn.name}")
                print(f"  参数: {json.dumps(args, ensure_ascii=False, indent=2)}")

            result = TOOLS[fn.name](**args) if fn.name in TOOLS else {"error": f"未知工具: {fn.name}"}

            if verbose:
                print(f"\n[Step {step+1}] 工具返回:")
                print(f"  {json.dumps(result, ensure_ascii=False, indent=2)}")

            tool_result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fn.name,
                    response={"result": result},
                )
            ))

        # 把工具结果作为 user turn 回传
        contents.append(types.Content(role="user", parts=tool_result_parts))

    return "Agent 未能完成推理，请检查工具配置。"


if __name__ == "__main__":
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
