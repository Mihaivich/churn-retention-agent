"""
tools/retrieve_cases.py

RAG 检索工具：根据当前客户特征，从知识库中检索相似的历史流失案例，
返回对应的挽留策略供 Agent 参考。

依赖：先运行 tools/build_knowledge_base.py 构建知识库。
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import Optional

# ── 配置（与 build_knowledge_base.py 保持一致）────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "churn_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────

# 单例：避免每次工具调用都重新加载模型（首次调用约需 2-3 秒）
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise RuntimeError(
                f"知识库目录不存在: {CHROMA_DIR}\n"
                "请先运行: python tools/build_knowledge_base.py"
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    return _collection


def retrieve_cases_tool(
    tenure: int,
    contract: str,
    internet_service: str,
    monthly_charges: float,
    payment_method: str = "Electronic check",
    risk_level: str = "high",
    n_results: int = 3,
) -> dict:
    """
    检索与当前客户特征最相似的历史流失案例及挽留策略。

    Agent 应在获取 churn_predict 结果后调用此工具，
    将检索到的历史策略融入最终挽留建议。

    Args:
        tenure:           在网月数
        contract:         合同类型
        internet_service: 网络服务类型
        monthly_charges:  月费
        payment_method:   付款方式
        risk_level:       来自 churn_predict 的风险等级
        n_results:        返回的相似案例数量（默认 3）

    Returns:
        dict: 包含相似案例列表和综合建议
    """
    # 把查询参数转成自然语言（与知识库文档格式对齐）
    query = (
        f"在网 {tenure} 个月，"
        f"合同类型为 {contract}，"
        f"网络服务为 {internet_service}，"
        f"月费 ${monthly_charges:.2f}，"
        f"付款方式为 {payment_method}。"
    )

    try:
        collection = _get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    cases = []
    all_strategies = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)  # 余弦距离转相似度
        strategy = meta.get("retention_strategy", "")
        all_strategies.append(strategy)

        cases.append({
            "similarity": similarity,
            "profile": {
                "tenure_months": meta.get("tenure"),
                "contract": meta.get("contract"),
                "internet_service": meta.get("internet_service"),
                "monthly_charges": meta.get("monthly_charges"),
                "payment_method": meta.get("payment_method"),
            },
            "retention_strategy": strategy,
            "document_excerpt": doc[:120] + "...",
        })

    # 汇总所有案例中出现的策略，去重后给 Agent 一个综合建议列表
    unique_strategies = []
    seen = set()
    for s in all_strategies:
        for item in s.split("；"):
            item = item.strip()
            if item and item not in seen:
                unique_strategies.append(item)
                seen.add(item)

    return {
        "success": True,
        "query_profile": {
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "risk_level": risk_level,
        },
        "similar_cases_found": len(cases),
        "cases": cases,
        "aggregated_strategies": unique_strategies,
        "summary": (
            f"找到 {len(cases)} 个相似历史案例，"
            f"综合推荐 {len(unique_strategies)} 项挽留策略。"
            f"最高相似度：{cases[0]['similarity']:.2%}。"
        ),
    }


# ── Anthropic tool_use schema ─────────────────────────────────────
TOOL_SCHEMA = {
    "name": "retrieve_cases",
    "description": (
        "从历史流失客户知识库中检索与当前客户最相似的案例及挽留策略。"
        "应在 churn_predict 之后调用，用于丰富挽留建议的依据。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tenure": {
                "type": "integer",
                "description": "客户在网月数",
            },
            "contract": {
                "type": "string",
                "enum": ["Month-to-month", "One year", "Two year"],
                "description": "合同类型",
            },
            "internet_service": {
                "type": "string",
                "enum": ["DSL", "Fiber optic", "No"],
                "description": "网络服务类型",
            },
            "monthly_charges": {
                "type": "number",
                "description": "月费金额",
            },
            "payment_method": {
                "type": "string",
                "description": "付款方式",
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "来自 churn_predict 的风险等级",
            },
            "n_results": {
                "type": "integer",
                "description": "返回相似案例数量，默认 3",
                "default": 3,
            },
        },
        "required": ["tenure", "contract", "internet_service", "monthly_charges"],
    },
}
