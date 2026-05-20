# Hugging Face Spaces — Churn Retention Agent
# Base: Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（chromadb 需要 build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements 利用 Docker layer 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# HF Spaces 要求使用 7860 端口，且以非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# 启动 Streamlit，绑定到 0.0.0.0:7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
