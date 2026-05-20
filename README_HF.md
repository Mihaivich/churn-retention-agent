---
title: Churn Retention Agent
emoji: 🔮
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Churn Retention Agent

AI-powered customer churn risk assessment & retention system built on a production MLOps pipeline.

## Architecture

```
Customer Input → ReAct Agent
                 ├── churn_predict  (scikit-learn model, local inference)
                 ├── retrieve_cases (ChromaDB + sentence-transformers RAG)
                 └── Gemini LLM     (retention script generation)
```

## Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn (Random Forest), trained on Telco Churn dataset |
| RAG | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Google Gemini 3.1 Flash Lite |
| Frontend | Streamlit |
| Data versioning | DVC |

## Environment Variables

Set `GEMINI_API_KEY` in Space Secrets for full functionality.  
Without it, the app runs in demo mode (risk prediction + RAG still work).
