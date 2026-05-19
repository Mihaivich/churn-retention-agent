"""
tools/build_knowledge_base.py

从 Telco 数据集构建 RAG 知识库。
筛选实际流失的客户，将每条记录转成自然语言文档，
用 sentence-transformers 生成 embedding，存入 ChromaDB。

运行（只需执行一次）:
    python tools/build_knowledge_base.py

输出:
    chroma_db/   本地向量数据库目录（由 .gitignore 排除）
"""

import json
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ── 路径配置 ──────────────────────────────────────────────────────
DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "churn_cases"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────


def record_to_document(row: pd.Series) -> str:
    """
    把一条客户记录转成自然语言文档。

    自然语言格式比结构化 JSON 的 embedding 效果更好，
    因为 sentence-transformers 是在自然语言语料上训练的。
    """
    contract_risk = {
        "Month-to-month": "高风险合同类型",
        "One year": "中等风险合同类型",
        "Two year": "低风险合同类型",
    }.get(row["Contract"], row["Contract"])

    services = []
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        if row.get(svc) == "Yes":
            services.append(svc)

    service_str = "、".join(services) if services else "无附加服务"

    doc = (
        f"客户档案：在网 {row['tenure']} 个月，"
        f"合同类型为 {row['Contract']}（{contract_risk}），"
        f"网络服务为 {row['InternetService']}，"
        f"月费 ${float(row['MonthlyCharges']):.2f}，"
        f"累计消费 ${float(row['TotalCharges']) if row['TotalCharges'] != ' ' else 0:.2f}，"
        f"付款方式为 {row['PaymentMethod']}，"
        f"附加服务：{service_str}。"
        f"客户性别 {row['gender']}，"
        f"{'老年用户' if row['SeniorCitizen'] == 1 else '非老年用户'}，"
        f"{'有伴侣' if row['Partner'] == 'Yes' else '无伴侣'}，"
        f"{'有家属' if row['Dependents'] == 'Yes' else '无家属'}。"
    )
    return doc


def build_retention_strategy(row: pd.Series) -> str:
    """
    根据客户特征生成对应的挽留策略标签。
    这作为文档的 metadata，让 Agent 检索后直接获得策略建议。
    """
    strategies = []

    if row["Contract"] == "Month-to-month":
        strategies.append("引导升级为年度合约，提供3个月折扣")
    if row["tenure"] < 12:
        strategies.append("新客关怀：专属客服跟进，赠送增值服务体验")
    if row["InternetService"] == "Fiber optic":
        strategies.append("光纤满意度回访，确认网速和稳定性达标")
    if row["PaymentMethod"] == "Electronic check":
        strategies.append("引导切换为信用卡自动扣款，降低流失摩擦")
    if float(row["MonthlyCharges"]) > 80:
        strategies.append("提供捆绑套餐，提升性价比感知")
    if row.get("OnlineSecurity") == "No" or row.get("TechSupport") == "No":
        strategies.append("推荐安全或技术支持附加服务，增加产品黏性")

    return "；".join(strategies) if strategies else "常规满意度维护"


def build_knowledge_base():
    print(f"读取数据集: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"总记录数: {len(df)}")

    # 只保留实际流失的客户作为"风险案例"
    churned = df[df["Churn"] == "Yes"].copy()
    print(f"流失客户数: {len(churned)}（将作为知识库案例）")

    # 初始化 ChromaDB（本地持久化）
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 使用 sentence-transformers embedding
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # 如果集合已存在则删除重建（确保数据一致）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"已删除旧集合: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # 余弦相似度
    )

    # 批量构建文档
    documents = []
    metadatas = []
    ids = []

    for idx, (_, row) in enumerate(churned.iterrows()):
        doc = record_to_document(row)
        strategy = build_retention_strategy(row)

        documents.append(doc)
        metadatas.append({
            "customer_id": row["customerID"],
            "tenure": int(row["tenure"]),
            "contract": row["Contract"],
            "internet_service": row["InternetService"],
            "monthly_charges": float(row["MonthlyCharges"]),
            "payment_method": row["PaymentMethod"],
            "churn": "Yes",
            "retention_strategy": strategy,
        })
        ids.append(f"case_{idx:04d}")

    # 分批写入（ChromaDB 推荐每批不超过 500 条）
    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )
        print(f"写入进度: {batch_end}/{total}")

    print(f"\n知识库构建完成！")
    print(f"集合名称: {COLLECTION_NAME}")
    print(f"文档总数: {collection.count()}")
    print(f"存储路径: {CHROMA_DIR}/")

    # 快速验证：用一个示例查询测试检索效果
    print("\n--- 验证检索（示例查询）---")
    test_query = "在网3个月，Month-to-month合同，Fiber optic，月费95元，电子支票付款"
    results = collection.query(query_texts=[test_query], n_results=2)
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n[结果 {i+1}] 相似度距离: {results['distances'][0][i]:.4f}")
        print(f"  文档: {doc[:80]}...")
        print(f"  挽留策略: {meta['retention_strategy']}")


if __name__ == "__main__":
    build_knowledge_base()
