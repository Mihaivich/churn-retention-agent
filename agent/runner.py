"""
agent/runner.py  — Google Gemini 版本 (google-genai SDK)

工具：
  1. churn_predict   — 预测流失概率
  2. retrieve_cases  — RAG 检索相似历史案例

运行前:
    pip install google-genai
    export GEMINI_API_KEY="AIza..."
    python tools/build_knowledge_base.py  # 首次需构建知识库

运行:
    python agent/runner.py
"""

import json
import os
import sys
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.churn_predict import churn_predict_tool, TOOL_SCHEMA as CHURN_SCHEMA
from tools.retrieve_cases import retrieve_cases_tool, TOOL_SCHEMA as RETRIEVE_SCHEMA

# ── 配置客户端 ────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3.1-flash-lite"

# ── 工具分发表 ────────────────────────────────────────────────────
TOOLS = {
    "churn_predict": churn_predict_tool,
    "retrieve_cases": retrieve_cases_tool,
}

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

# 两个工具合并进同一个 Tool 对象（Gemini 推荐的做法）
GEMINI_TOOLS = [types.Tool(function_declarations=[
    build_gemini_tool(CHURN_SCHEMA).function_declarations[0],
    build_gemini_tool(RETRIEVE_SCHEMA).function_declarations[0],
])]

SYSTEM_PROMPT = """你是一名电信公司的客户留存专家 Agent。

你的标准工作流程（每次必须完整执行两步）：
1. 调用 churn_predict 评估客户流失风险，获取概率、风险等级和风险因素
2. 调用 retrieve_cases 检索相似历史流失案例，获取经过验证的挽留策略
3. 综合两个工具的结果，给出结构化挽留建议

输出格式：
- 风险评估：概率、等级、核心风险因素
- 历史参考：相似案例数、最高相似度、历史成功策略
- 行动建议：按优先级排列，每条对应具体风险因素或历史案例
- 沟通话术：1-2 句可直接使用的开场白

原则：
- 高风险（≥70%）立即介入；中风险（40-70%）近期跟进；低风险（<40%）常规维护
- 所有建议必须基于工具返回的数据，不得凭空生成
- retrieve_cases 的 risk_level 参数使用 churn_predict 返回的值"""


def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    ReAct 循环：generate_content → function_call → tool result → final answer
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

    for step in range(6):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
            ),
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

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

        # 执行工具，收集结果
        tool_result_parts = []
        for fn in fn_calls:
            args = dict(fn.args)
            if verbose:
                print(f"\n[Step {step+1}] 调用工具: {fn.name}")
                print(f"  参数: {json.dumps(args, ensure_ascii=False, indent=2)}")

            result = TOOLS[fn.name](**args) if fn.name in TOOLS else {"error": f"未知工具: {fn.name}"}

            if verbose:
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                if len(result_str) > 800:
                    result_str = result_str[:800] + "\n  ... (truncated)"
                print(f"\n[Step {step+1}] 工具返回:\n  {result_str}")

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