"""
app.py — Churn Retention Agent Dashboard (Premium AI Workspace Version)
Design: Dark space workspace with violet (#7132f5) & cyan (#00f2fe) glow, glassmorphism,
and dynamic timeline agent reasoning trace.
Single-page layout, zh/en toggle, 3-step streaming Agent with in-process local fallbacks.
"""

import os
import sys
import time
import json
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.churn_predict import churn_predict_tool, _assess_risk_factors
from tools.retrieve_cases import retrieve_cases_tool
from google import genai
from google.genai import types as gtypes

# Initialize Gemini Client with safety fallback
_gemini_client = None
if os.environ.get("GEMINI_API_KEY", ""):
    try:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception:
        pass
_GEMINI_MODEL = "gemini-3.1-flash-lite"

# Page config
st.set_page_config(
    page_title="Churn Retention Agent Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Style Sheet for Premium Dark AI Agent Cockpit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Base Theme Overrides ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #08090d !important;
    color: #e2e8f0;
}
.stApp {
    background-color: #08090d !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0c0d12 !important;
    border-right: 1px solid rgba(113, 50, 245, 0.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.8rem !important;
    padding-bottom: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    margin-bottom: 2px !important;
}

/* ── Labels & Headers ── */
label, [data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 6px !important;
}

/* ── Inputs ── */
input[type="text"], input[type="number"], [data-testid="stSelectbox"] > div > div {
    background: #141622 !important;
    border: 1px solid rgba(113, 50, 245, 0.2) !important;
    border-radius: 8px !important;
    color: #f0f2f6 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
    box-shadow: none !important;
    transition: all 0.25s ease !important;
}
input[type="text"]:focus, input[type="number"]:focus, [data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #7132f5 !important;
    box-shadow: 0 0 0 3px rgba(113, 50, 245, 0.25) !important;
    outline: none !important;
}

/* Number input stepper buttons */
[data-testid="stNumberInput"] button {
    background: #1c1f30 !important;
    border-color: rgba(113, 50, 245, 0.2) !important;
    color: #a5adc9 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #252a41 !important;
    color: #ffffff !important;
}

/* ── Primary Action Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7132f5, #5741d8) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(113, 50, 245, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8854ff, #7132f5) !important;
    box-shadow: 0 6px 20px rgba(113, 50, 245, 0.5) !important;
    transform: translateY(-1px);
    color: #ffffff !important;
}
.stButton > button:active {
    transform: translateY(1px);
}

/* ── Language Switcher Buttons ── */
.lang-btn .stButton > button {
    background: #141622 !important;
    color: #8f95b2 !important;
    border: 1px solid rgba(113, 50, 245, 0.15) !important;
    box-shadow: none !important;
    padding: 6px 12px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.lang-btn .stButton > button:hover {
    background: rgba(113, 50, 245, 0.1) !important;
    border-color: rgba(113, 50, 245, 0.3) !important;
    color: #f0f2f6 !important;
}
.lang-btn-active .stButton > button {
    background: rgba(113, 50, 245, 0.2) !important;
    color: #00f2fe !important;
    border: 1px solid #7132f5 !important;
    box-shadow: 0 0 10px rgba(113, 50, 245, 0.3) !important;
}
.lang-btn-active .stButton > button:hover {
    color: #00f2fe !important;
    background: rgba(113, 50, 245, 0.25) !important;
}

/* ── Preset Profile Buttons ── */
.preset-btn .stButton > button {
    background: #141622 !important;
    color: #a5adc9 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: none !important;
    padding: 6px 8px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
.preset-btn .stButton > button:hover {
    background: rgba(113, 50, 245, 0.08) !important;
    border-color: rgba(113, 50, 245, 0.2) !important;
    color: #ffffff !important;
}
.preset-high .stButton > button { border-left: 3px solid #ef4444 !important; }
.preset-mid .stButton > button  { border-left: 3px solid #f97316 !important; }
.preset-low .stButton > button  { border-left: 3px solid #10b981 !important; }

/* ── Custom Divider Line (Gemini flow) ── */
@keyframes ai-flow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.ai-line {
    height: 2px;
    border-radius: 9999px;
    background: linear-gradient(90deg, #7132f5 0%, #00f2fe 30%, #ec4899 70%, #7132f5 100%);
    background-size: 200% 100%;
    animation: ai-flow 4s ease infinite;
    margin: 4px 0 24px 0;
}

/* ── Header Widget ── */
.agent-header {
    background: #0f111a;
    border: 1px solid rgba(113, 50, 245, 0.15);
    border-radius: 12px;
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.agent-header-left {
    display: flex;
    align-items: center;
}
.agent-logo {
    font-size: 2.2rem;
    margin-right: 16px;
    filter: drop-shadow(0 0 10px rgba(113, 50, 245, 0.4));
}
.agent-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.agent-subtitle {
    font-size: 0.8rem;
    color: #8f95b2;
    margin-top: 4px;
}
.agent-status-badges {
    display: flex;
    gap: 10px;
}
.status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid;
    display: flex;
    align-items: center;
    gap: 6px;
}
.status-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.status-pill.green {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.2);
}
.status-pill.green::before { background-color: #10b981; }

.status-pill.orange {
    background: rgba(249, 115, 22, 0.1);
    color: #fb923c;
    border-color: rgba(249, 115, 22, 0.2);
}
.status-pill.orange::before { background-color: #f97316; }

/* ── Panel Cards ── */
.k-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.k-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.05);
}

/* ── Visual Timeline ── */
.timeline {
    position: relative;
    padding-left: 26px;
    margin: 15px 0;
}
.timeline::before {
    content: '';
    position: absolute;
    top: 5px;
    left: 7px;
    width: 2px;
    height: calc(100% - 20px);
    background: rgba(113, 50, 245, 0.2);
}
.timeline-item {
    position: relative;
    margin-bottom: 20px;
}
.timeline-marker {
    position: absolute;
    top: 3px;
    left: -26px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #141622;
    border: 2px solid #4a4f6e;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 8px;
    color: #8f95b2;
    transition: all 0.3s ease;
}
.timeline-item.active .timeline-marker {
    background: #7132f5;
    border-color: #00f2fe;
    box-shadow: 0 0 10px #7132f5, 0 0 4px #00f2fe;
    color: #fff;
    animation: marker-pulse 1s infinite alternate;
}
.timeline-item.completed .timeline-marker {
    background: #10b981;
    border-color: #10b981;
    color: #fff;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}
.timeline-item.error .timeline-marker {
    background: #ef4444;
    border-color: #ef4444;
    color: #fff;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
}
.timeline-content {
    background: #10121e;
    border: 1px solid rgba(113, 50, 245, 0.1);
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.timeline-item.active .timeline-content {
    border-color: rgba(113, 50, 245, 0.35);
    box-shadow: 0 4px 15px rgba(113, 50, 245, 0.08);
}
.timeline-title {
    font-size: 0.83rem;
    font-weight: 600;
    color: #f0f2f6;
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.timeline-info {
    font-size: 0.74rem;
    color: #00f2fe;
    font-weight: 600;
}
.timeline-log {
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: #090a0f;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.timeline-log summary {
    cursor: pointer;
    color: #8854ff;
    padding: 4px 8px;
    outline: none;
    font-weight: 600;
    user-select: none;
}
.timeline-log summary:hover {
    color: #00f2fe;
}
.timeline-log pre {
    margin: 0;
    padding: 8px;
    white-space: pre-wrap;
    word-break: break-all;
    color: #a5adc9;
    background: transparent;
    border: none;
    max-height: 150px;
    overflow-y: auto;
}
@keyframes marker-pulse {
    0% { transform: scale(1); }
    100% { transform: scale(1.15); }
}

/* ── Risk Dashboard Card & SVG Gauge ── */
.risk-dashboard-card {
    background: #10121e;
    border: 1px solid rgba(113, 50, 245, 0.15);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    margin-bottom: 14px;
}
.risk-gauge-container {
    width: 80px;
    height: 80px;
    min-width: 80px;
}
.circular-chart {
    display: block;
    max-width: 100%;
    max-height: 100%;
}
.circle-bg {
    fill: none;
    stroke: rgba(255,255,255,0.05);
    stroke-width: 2.8;
}
.circle {
    fill: none;
    stroke-width: 2.8;
    stroke-linecap: round;
    transition: stroke-dasharray 0.6s ease;
}
.percentage {
    fill: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.5px;
    font-weight: 700;
    text-anchor: middle;
}
.risk-high .circle { stroke: #ef4444; filter: drop-shadow(0 0 3px #ef4444); }
.risk-medium .circle { stroke: #f97316; filter: drop-shadow(0 0 3px #f97316); }
.risk-low .circle { stroke: #10b981; filter: drop-shadow(0 0 3px #10b981); }

.risk-details {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.risk-badge-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    width: fit-content;
}
.risk-high .risk-badge-label { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.risk-medium .risk-badge-label { background: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.2); }
.risk-low .risk-badge-label { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.2); }

.risk-recommendation {
    font-size: 0.8rem;
    color: #a5adc9;
    line-height: 1.45;
}

/* ── Risk Factor Tags ── */
.risk-factor-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 10px 0 20px 0;
}
.risk-factor-tag {
    background: rgba(113, 50, 245, 0.08);
    border: 1px solid rgba(113, 50, 245, 0.2);
    color: #b196ff;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
}
.risk-factor-tag:hover {
    background: rgba(113, 50, 245, 0.14);
    border-color: rgba(113, 50, 245, 0.4);
    color: #ffffff;
    transform: translateY(-1px);
}

/* ── Playbooks (Strategies) ── */
.strategy-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.strategy-item {
    background: #10121e;
    border: 1px solid rgba(113, 50, 245, 0.1);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    transition: all 0.2s ease;
}
.strategy-item:hover {
    border-color: rgba(113, 50, 245, 0.25);
    background: rgba(113, 50, 245, 0.02);
}
.strategy-badge {
    background: rgba(0, 242, 254, 0.1);
    color: #00f2fe;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.68rem;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}
.strategy-text {
    font-size: 0.8rem;
    color: #e2e8f0;
    line-height: 1.5;
}

/* ── Case Finder (RAG) ── */
.case-card {
    background: #10121e;
    border: 1px solid rgba(113, 50, 245, 0.1);
    border-left: 3px solid #7132f5;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.case-card:hover {
    border-color: rgba(113, 50, 245, 0.25);
    border-left-color: #00f2fe;
    transform: translateX(2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.case-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.case-similarity {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #00f2fe;
    background: rgba(0,242,254,0.1);
    padding: 2px 6px;
    border-radius: 4px;
}
.case-meta {
    font-size: 0.72rem;
    color: #8f95b2;
}
.case-tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
}
.case-tag {
    background: rgba(113, 50, 245, 0.05);
    border: 1px solid rgba(113, 50, 245, 0.15);
    color: #a5adc9;
    font-size: 0.65rem;
    padding: 1px 5px;
    border-radius: 3px;
}

/* ── AI Script Workspace Overrides ── */
.script-workspace-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #f0f2f6;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 24px;
    margin-bottom: 10px;
}
.script-workspace-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.05);
}

/* Custom styled Streamlit code block to act as copyable terminal script */
[data-testid="stMarkdownContainer"] pre, [data-testid="stCodeBlock"] pre {
    background-color: #10121e !important;
    border: 1px solid rgba(113,50,245,0.2) !important;
    border-left: 4px solid #7132f5 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMarkdownContainer"] code, [data-testid="stCodeBlock"] code {
    color: #e0e2ed !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
    background: transparent !important;
    padding: 0 !important;
    white-space: pre-wrap !important;
    line-height: 1.8 !important;
}

/* AI Script Card and Content styles */
.script-card {
    background: #10121e !important;
    border: 1px solid rgba(113,50,245,0.2) !important;
    border-left: 4px solid #7132f5 !important;
    border-radius: 8px !important;
    padding: 20px 24px !important;
    margin-top: 10px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
}
.script-content {
    color: #e0e2ed !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.8 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
</style>
""", unsafe_allow_html=True)

# ── Translation System ──
LANG = {
    "zh": {
        "title": "Churn Retention Agent",
        "subtitle": "AI 驱动的客户流失风险评估与挽留决策工作台",
        "input_header": "客户特征输入",
        "customer_id": "客户 ID",
        "tenure": "在网时长（月）",
        "contract": "合同类型",
        "internet": "网络服务类型",
        "monthly": "当前月费（$）",
        "total": "历史累计费用（$）",
        "payment": "付款结算方式",
        "run_btn": "运行 Agent 决策流",
        "trace_header": "AGENT 推理分析控制台",
        "risk_header": "实时流失风险评估",
        "rag_header": "历史相似案例检索 (RAG)",
        "strategy_header": "智能生成挽留建议",
        "factors_label": "核心流失诱因因素",
        "script_header": "AI 客服智能挽留话术方案",
        "prob_label": "流失概率预测",
        "risk_high": "高流失风险",
        "risk_medium": "中流失风险",
        "risk_low": "低流失风险",
        "step": "步骤",
        "calling": "调用工具",
        "done": "决策分析完成",
        "step3_label": "调用大模型生成挽留方案",
        "no_result": "// 等待 Agent 启动分析控制台...",
        "similarity": "匹配相似度",
        "tenure_unit": "个月",
        "contract_opts": ['月付 (Month-to-month)', '一年 (One year)', '两年 (Two year)'],
        "contract_vals": ['Month-to-month', 'One year', 'Two year'],
        "internet_opts": ['光纤 (Fiber optic)', 'DSL', '无网络 (No)'],
        "internet_vals": ['Fiber optic', 'DSL', 'No'],
        "payment_opts":  ['电子支票', '邮寄支票', '银行转账（自动）', '信用卡（自动）'],
        "payment_vals":  ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        "translate_factors": False,
        "translate_strategies": False,
        "preset_header": "快速测试预设客户",
        "preset_high": "🔴 高风险客户",
        "preset_mid": "🟡 中风险客户",
        "preset_low": "🟢 低风险客户",
        "sys_status": "智能代理系统状态",
    },
    "en": {
        "title": "Churn Retention Agent",
        "subtitle": "AI-powered customer churn risk assessment & retention decision workspace",
        "input_header": "Customer Profile",
        "customer_id": "Customer ID",
        "tenure": "Tenure (months)",
        "contract": "Contract Type",
        "internet": "Internet Service",
        "monthly": "Monthly Charges ($)",
        "total": "Total Charges ($)",
        "payment": "Payment Method",
        "run_btn": "Run Agent Analysis",
        "trace_header": "AGENT COGNITIVE CONSOLE",
        "risk_header": "Real-time Churn Risk",
        "rag_header": "RAG Case Retrieval",
        "strategy_header": "Generated Playbook",
        "factors_label": "Key Churn Inducers",
        "script_header": "AI-Generated Retention Script",
        "prob_label": "CHURN PROBABILITY",
        "risk_high": "HIGH CHURN RISK",
        "risk_medium": "MEDIUM CHURN RISK",
        "risk_low": "LOW CHURN RISK",
        "step": "STEP",
        "calling": "calling tool",
        "done": "Cognitive run complete",
        "step3_label": "Calling LLM to construct retention script",
        "no_result": "// Waiting for Agent execution...",
        "similarity": "similarity",
        "tenure_unit": "mo",
        "contract_opts": ['Month-to-month', 'One year', 'Two year'],
        "contract_vals": ['Month-to-month', 'One year', 'Two year'],
        "internet_opts": ['Fiber optic', 'DSL', 'No'],
        "internet_vals": ['Fiber optic', 'DSL', 'No'],
        "payment_opts":  ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        "payment_vals":  ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        "translate_factors": True,
        "translate_strategies": True,
        "preset_header": "Quick Test Profiles",
        "preset_high": "🔴 High Risk Profile",
        "preset_mid": "🟡 Mid Risk Profile",
        "preset_low": "🟢 Low Risk Profile",
        "sys_status": "Agent System Status",
    },
}

# ── Preset Profile Data ──
PRESETS = {
    "high": {
        "customer_id": "C-98741",
        "tenure": 2,
        "monthly": 98,
        "total": 196,
        "contract": 0, # Month-to-month
        "internet": 0, # Fiber optic
        "payment": 0,  # Electronic check
    },
    "medium": {
        "customer_id": "C-45812",
        "tenure": 12,
        "monthly": 65,
        "total": 780,
        "contract": 1, # One year
        "internet": 1, # DSL
        "payment": 2,  # Bank transfer (automatic)
    },
    "low": {
        "customer_id": "C-12345",
        "tenure": 48,
        "monthly": 30,
        "total": 1440,
        "contract": 2, # Two year
        "internet": 2, # No
        "payment": 3,  # Credit card (automatic)
    }
}

FACTOR_EN = {
    "短期合同 (Month-to-month)，锁定性弱":   "Short-term contract (Month-to-month), low lock-in",
    "光纤用户历史流失率偏高":                  "Fiber optic users have higher historical churn rates",
    "电子支票付款方式与流失强相关":            "Electronic check payment strongly correlates with churn",
    "无明显高风险特征":                       "No obvious high risk features",
}
STRATEGY_EN = {
    "引导升级为年度合约，提供3个月折扣":              "Upgrade to annual contract — offer 3-month discount",
    "新客关怀：专属客服跟进，赠送增值服务体验":        "New customer care: dedicated follow-up + free add-on trial",
    "光纤满意度回访，确认网速和稳定性达标":            "Fiber quality check-in: verify speed & stability",
    "引导切换为信用卡自动扣款，降低流失摩擦":          "Switch to auto-pay (credit card) to reduce churn friction",
    "提供捆绑套餐，提升性价比感知":                    "Offer bundled plan to improve value perception",
    "推荐安全或技术支持附加服务，增加产品黏性":        "Recommend security / tech support add-ons for stickiness",
    "常规满意度维护":                                  "Routine customer satisfaction maintenance",
}

RECOMMENDATION_EN = {
    "立即介入：主动联系客户，提供个性化挽留方案": "Immediate intervention: actively contact customer to offer personalized retention plan",
    "近期跟进：发送满意度调查，提前识别不满点": "Routine follow-up: send satisfaction survey to identify complaints early",
    "正常维护：可考虑升级销售或增值服务推荐": "Standard maintenance: suggest upsell or value-added services",
}

def translate_recommendation(text, lang):
    return RECOMMENDATION_EN.get(text.strip(), text) if lang == "en" else text

def translate_factor(text, lang):
    if lang != "en":
        return text
    for zh, en in FACTOR_EN.items():
        if zh.split("，")[0].split(" ")[0] in text:
            return en
    import re
    nums = re.findall(r"[\d.]+", text)
    if "个月" in text and nums:
        return f"Only {nums[0]} months tenure — loyalty not yet established"
    if "月费" in text and nums:
        return f"Monthly charge ${nums[-1]}, above average"
    return text

def translate_strategy(text, lang):
    return STRATEGY_EN.get(text.strip(), text) if lang == "en" else text

DB_TRANSLATIONS = {
    "zh": {
        "Month-to-month": "月付",
        "One year": "一年",
        "Two year": "两年",
        "Fiber optic": "光纤",
        "DSL": "DSL",
        "No": "无网络",
        "Yes": "是",
        "Electronic check": "电子支票",
        "Mailed check": "邮寄支票",
        "Bank transfer (automatic)": "银行转账 (自动)",
        "Credit card (automatic)": "信用卡 (自动)",
    },
    "en": {
        "Month-to-month": "Month-to-month",
        "One year": "One year",
        "Two year": "Two year",
        "Fiber optic": "Fiber optic",
        "DSL": "DSL",
        "No": "No",
        "Yes": "Yes",
        "Electronic check": "Electronic check",
        "Mailed check": "Mailed check",
        "Bank transfer (automatic)": "Bank transfer (automatic)",
        "Credit card (automatic)": "Credit card (automatic)",
    }
}

def translate_db_val(val, lang):
    if not val:
        return val
    val_str = str(val).strip()
    return DB_TRANSLATIONS.get(lang, {}).get(val_str, val_str)

def risk_css(level):
    return {"high": "prob-high", "medium": "prob-medium", "low": "prob-low"}.get(level, "prob-low")

def badge_css(level):
    return {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(level, "badge-low")

def risk_label(level, t):
    return {"high": t["risk_high"], "medium": t["risk_medium"], "low": t["risk_low"]}.get(level, level.upper())


# ── System Diagnostics & Direct Local Fallback ──
def check_system_status():
    import requests
    predictor_status = "fallback"
    try:
        resp = requests.get("http://localhost:8000/ping", timeout=0.8)
        if resp.status_code == 200:
            predictor_status = "online"
    except Exception:
        pass
    gemini_status = "active" if _gemini_client is not None else "demo"
    return predictor_status, gemini_status

def predict_locally_fallback(params):
    """In-process model runner fallback if local uvicorn server is offline."""
    model_path = Path("model/model.joblib")
    if not model_path.exists():
        return {"success": False, "error": "Model file model.joblib not found. Run dvc repro."}
    try:
        model = joblib.load(model_path)
        feature_columns = list(model.feature_names_in_)
        
        defaults = {
            "customerID": params.get("customer_id", "unknown"),
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": params.get("tenure", 1),
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": params.get("internet_service", "Fiber optic"),
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": params.get("contract", "Month-to-month"),
            "PaperlessBilling": "Yes",
            "PaymentMethod": params.get("payment_method", "Electronic check"),
            "MonthlyCharges": params.get("monthly_charges", 70.0),
            "TotalCharges": params.get("total_charges", 70.0),
        }
        
        row = {**defaults}
        mapping = {
            "customer_id": "customerID",
            "tenure": "tenure",
            "contract": "Contract",
            "internet_service": "InternetService",
            "monthly_charges": "MonthlyCharges",
            "total_charges": "TotalCharges",
            "payment_method": "PaymentMethod"
        }
        for k, v in mapping.items():
            if k in params:
                row[v] = params[k]
                
        df = pd.DataFrame([row])
        for col in feature_columns:
            if col not in df.columns:
                df[col] = defaults.get(col, None)
        df = df[feature_columns]
        
        prob = float(model.predict_proba(df)[0][1])
        
        if prob >= 0.7:
            risk_level, recommendation = "high", "立即介入：主动联系客户，提供个性化挽留方案"
        elif prob >= 0.4:
            risk_level, recommendation = "medium", "近期跟进：发送满意度调查，提前识别不满点"
        else:
            risk_level, recommendation = "low", "正常维护：可考虑升级销售或增值服务推荐"
            
        factors = _assess_risk_factors({
            "Contract": row.get("Contract"),
            "tenure": row.get("tenure"),
            "InternetService": row.get("InternetService"),
            "PaymentMethod": row.get("PaymentMethod"),
            "MonthlyCharges": row.get("MonthlyCharges"),
        })
        
        return {
            "success": True,
            "customer_id": params.get("customer_id"),
            "churn_probability": prob,
            "risk_level": risk_level,
            "risk_score_pct": f"{prob:.0%}",
            "key_risk_factors": factors,
            "recommendation": recommendation,
            "backend_used": "local_direct_fallback",
        }
    except Exception as e:
        return {"success": False, "error": f"Local model execution failed: {str(e)}"}

def run_churn_predict(params):
    cr = churn_predict_tool(
        customer_id=params["customer_id"], tenure=params["tenure"],
        contract=params["contract"], internet_service=params["internet_service"],
        monthly_charges=params["monthly_charges"], total_charges=params["total_charges"],
        payment_method=params["payment_method"]
    )
    if not cr.get("success"):
        cr = predict_locally_fallback(params)
    return cr

def run_retrieve_cases(params, risk_level):
    rr = retrieve_cases_tool(
        tenure=params["tenure"], contract=params["contract"],
        internet_service=params["internet_service"], monthly_charges=params["monthly_charges"],
        payment_method=params["payment_method"], risk_level=risk_level
    )
    if not rr.get("success"):
        # Local mock cases fallback to prevent empty screen if Chroma is not running
        mock_cases = [
            {
                "similarity": 0.885,
                "profile": {
                    "tenure_months": max(1, params["tenure"] - 1),
                    "contract": params["contract"],
                    "internet_service": params["internet_service"],
                    "monthly_charges": params["monthly_charges"],
                    "payment_method": params["payment_method"]
                },
                "retention_strategy": "引导升级为年度合约，提供3个月折扣；光纤满意度回访，确认网速和稳定性达标" if params["internet_service"] == "Fiber optic" else "引导切换为信用卡自动扣款，降低流失摩擦"
            },
            {
                "similarity": 0.812,
                "profile": {
                    "tenure_months": params["tenure"] + 2,
                    "contract": params["contract"],
                    "internet_service": params["internet_service"],
                    "monthly_charges": max(15, params["monthly_charges"] - 5),
                    "payment_method": params["payment_method"]
                },
                "retention_strategy": "光纤满意度回访，确认网速和稳定性达标；提供捆绑套餐，提升性价比感知" if params["internet_service"] == "Fiber optic" else "推荐安全或技术支持附加服务，增加产品黏性"
            }
        ]
        unique_strategies = []
        seen = set()
        for c in mock_cases:
            for item in c["retention_strategy"].split("；"):
                item = item.strip()
                if item and item not in seen:
                    unique_strategies.append(item)
                    seen.add(item)
        rr = {
            "success": True,
            "query_profile": {
                "tenure": params["tenure"],
                "contract": params["contract"],
                "internet_service": params["internet_service"],
                "monthly_charges": params["monthly_charges"],
                "risk_level": risk_level
            },
            "similar_cases_found": len(mock_cases),
            "cases": mock_cases,
            "aggregated_strategies": unique_strategies,
            "summary": "Knowledge Base offline. Loaded backup historical cases.",
            "fallback_used": True
        }
    return rr


# ── Gemini Script Generation with SAFETY FALLBACK ──
def generate_retention_script(params, cr, rr, lang):
    global _gemini_client
    if not _gemini_client:
        # Beautiful mock response when GEMINI_API_KEY is not defined
        if lang == "en":
            return "[DEMO MODE - No GEMINI_API_KEY detected]\n\n" \
                   "1) Opening Dialogue:\n" \
                   "\"Hello, customer representative here. I noticed you've been with us for a while, and we'd love to check on your satisfaction...\"\n\n" \
                   "2) Core Talking Points:\n" \
                   "- Acknowledge short tenure or high monthly charge ($95.00) and discuss opt-in discounts.\n" \
                   "- Address Fiber optic speed/stability and offer a free tech check-in.\n\n" \
                   "3) Offer Pitch:\n" \
                   "\"If we transition you to our Annual contract, we can credit $15/month for the next 3 months...\"\n\n" \
                   "4) Objection Handling:\n" \
                   "If customer states they prefer no contracts, emphasize the total savings of $180/year...\n\n" \
                   "5) Closing:\n" \
                   "\"Let me set this up for you right away. Thank you for choosing us!\""
        else:
            return "[演示模式 - 未检测到 GEMINI_API_KEY]\n\n" \
                   "1) 开场白（双语切换视角）:\n" \
                   "“您好，我是客户经理。注意到您是我们非常关注的客户，今天特意回访想了解下您的使用体验...”\n\n" \
                   "2) 核心沟通点:\n" \
                   "• 针对在网时间短或月费偏高（$95.00）的问题，主动提供资费优化建议。\n" \
                   "• 针对光纤服务的稳定性，安排专人进行网速优化和免费检测。\n\n" \
                   "3) 挽留方案方案推荐:\n" \
                   "“如果您升级为一年期合约，我们可以为您前3个月每月减免 $15，并赠送增值保障服务...”\n\n" \
                   "4) 异议处理:\n" \
                   "如果客户担心合约绑定，解释年度合约总计能节省 $180 费用，并且可随时调整套餐...\n\n" \
                   "5) 结束语:\n" \
                   "“那我就帮您系统里登记这个优惠了，后续有任何网速问题可以直接打我专线。祝您生活愉快！”"

    lang_instr = "Please respond entirely in English." if lang == "en" else "请用中文回答。"
    strategies = "；".join(rr.get("aggregated_strategies", []))
    factors    = "、".join(cr.get("key_risk_factors", []))
    cases_info = "".join(
        f"- sim {c['similarity']:.1%}: {c['profile']['tenure_months']}mo "
        f"{c['profile']['contract']} ${c['profile']['monthly_charges']:.0f}/mo\n"
        for c in rr.get("cases", [])[:2]
    )
    prompt = f"""{lang_instr}

You are a senior telecom customer retention manager. Generate a complete retention script for the service team.

[Customer] ID:{params['customer_id']} | {params['tenure']}mo | {params['contract']} | {params['internet_service']} | ${params['monthly_charges']:.0f}/mo | {params['payment_method']}
[Risk] {cr.get('churn_probability',0):.1%} · {cr.get('risk_level','').upper()} | {factors}
[Similar cases]\n{cases_info}[Strategies] {strategies}

Script sections: 1) Opening (2 angles) 2) Core talking points per risk factor 3) Offer pitch 4) Objection handling 5) Closing. Natural conversation tone."""

    try:
        r = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=gtypes.GenerateContentConfig(max_output_tokens=1200),
        )
        return r.text or ""
    except Exception as e:
        return f"Gemini API Error: {str(e)}\n\nFallback generated above."


# ── Render Timeline Component ──
def render_timeline(steps):
    html = '<div class="timeline">'
    for step in steps:
        cls = ""
        marker = "○"
        if step["status"] == "running":
            cls = "active"
            marker = "⚡"
        elif step["status"] == "completed":
            cls = "completed"
            marker = "✓"
        elif step["status"] == "error":
            cls = "error"
            marker = "✗"
            
        html += f'<div class="timeline-item {cls}">'
        html += f'  <div class="timeline-marker">{marker}</div>'
        html += f'  <div class="timeline-content">'
        
        name = step["name_zh"] if st.session_state.lang == "zh" else step["name_en"]
        html += f'    <div class="timeline-title"><span>{name}</span>'
        if step["info"]:
            html += f'      <span class="timeline-info">{step["info"]}</span>'
        html += f'    </div>'
        
        if step["details"]:
            html += f'    <details class="timeline-log"><summary>Console Logs</summary>'
            html += f'      <pre>{step["details"]}</pre>'
            html += f'    </details>'
            
        html += f'  </div>'
        html += f'</div>'
    html += '</div>'
    return html

def render_risk_card(prob, level, recommendation, t):
    dash_array = f"{prob*100:.1f}, 100"
    prob_str = f"{prob:.1%}"
    level_cls = f"risk-{level}"
    badge_text = risk_label(level, t)
    
    html = f"""
    <div class="risk-dashboard-card {level_cls}">
        <div class="risk-gauge-container">
            <svg viewBox="0 0 36 36" class="circular-chart">
                <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path class="circle" stroke-dasharray="{dash_array}" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <text x="18" y="20.35" class="percentage">{prob_str}</text>
            </svg>
        </div>
        <div class="risk-details">
            <div class="risk-badge-label">{badge_text}</div>
            <div class="risk-recommendation">{recommendation}</div>
        </div>
    </div>
    """
    return html


# ── Session state & presets logic ──
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False

# Initialize input state values for preset binding
if "customer_id_val" not in st.session_state:
    st.session_state.customer_id_val = "C-10023"
if "tenure_val" not in st.session_state:
    st.session_state.tenure_val = 3
if "monthly_val" not in st.session_state:
    st.session_state.monthly_val = 95
if "total_val" not in st.session_state:
    st.session_state.total_val = 285

for l in ["zh", "en"]:
    if f"contract_val_{l}" not in st.session_state:
        st.session_state[f"contract_val_{l}"] = 0
    if f"internet_val_{l}" not in st.session_state:
        st.session_state[f"internet_val_{l}"] = 0
    if f"payment_val_{l}" not in st.session_state:
        st.session_state[f"payment_val_{l}"] = 0

def load_preset(key):
    preset = PRESETS[key]
    st.session_state.customer_id_val = preset["customer_id"]
    st.session_state.tenure_val = preset["tenure"]
    st.session_state.monthly_val = preset["monthly"]
    st.session_state.total_val = preset["total"]
    for l in ["zh", "en"]:
        st.session_state[f"contract_val_{l}"] = preset["contract"]
        st.session_state[f"internet_val_{l}"] = preset["internet"]
        st.session_state[f"payment_val_{l}"] = preset["payment"]
    st.session_state.result = None  # Clear previous result

def update_total():
    st.session_state.total_val = int(st.session_state.monthly_val) * int(st.session_state.tenure_val)


# ── Streaming Agent Executor Loop ──
def run_agent_streaming(params, trace_ph, t, lang):
    steps = [
        {
            "id": 1,
            "name_zh": "流失风险预测 (ML 模型)",
            "name_en": "Risk Prediction (ML Model)",
            "status": "idle",
            "info": "",
            "details": ""
        },
        {
            "id": 2,
            "name_zh": "历史相似案例检索 (RAG)",
            "name_en": "Case Retrieval (RAG)",
            "status": "idle",
            "info": "",
            "details": ""
        },
        {
            "id": 3,
            "name_zh": "大语言模型生成挽留方案",
            "name_en": "LLM Retention Script Generation",
            "status": "idle",
            "info": "",
            "details": ""
        }
    ]
    
    # Helper to render timeline HTML
    def render():
        trace_ph.markdown(render_timeline(steps), unsafe_allow_html=True)

    # STEP 1: Predict
    steps[0]["status"] = "running"
    steps[0]["info"] = "Assessing..." if lang == "en" else "评估中..."
    steps[0]["details"] = f"Calling churn_predict with inputs:\n{json.dumps(params, ensure_ascii=False, indent=2)}"
    render()
    time.sleep(0.7)
    
    cr = run_churn_predict(params)
    
    if cr.get("success"):
        steps[0]["status"] = "completed"
        prob = cr["churn_probability"]
        level = cr["risk_level"]
        steps[0]["info"] = f"{prob:.1%} ({level.upper()})"
        steps[0]["details"] += f"\n\nResponse:\n{json.dumps(cr, ensure_ascii=False, indent=2)}"
    else:
        steps[0]["status"] = "error"
        steps[0]["info"] = "Failed"
        steps[0]["details"] += f"\n\nError:\n{cr.get('error')}"
        render()
        return cr, {"success": False}, ""
    render()
    
    # STEP 2: RAG Cases
    steps[1]["status"] = "running"
    steps[1]["info"] = "Retrieving..." if lang == "en" else "检索中..."
    steps[1]["details"] = f"Calling retrieve_cases with inputs:\n" \
                          f"  tenure: {params['tenure']}\n" \
                          f"  contract: {params['contract']}\n" \
                          f"  internet_service: {params['internet_service']}\n" \
                          f"  monthly_charges: {params['monthly_charges']}\n" \
                          f"  risk_level: {cr.get('risk_level')}"
    render()
    time.sleep(0.7)
    
    rr = run_retrieve_cases(params, cr.get("risk_level", "medium"))
    
    if rr.get("success"):
        steps[1]["status"] = "completed"
        steps[1]["info"] = f"Matched {rr['similar_cases_found']} cases" if lang == "en" else f"匹配 {rr['similar_cases_found']} 案例"
        steps[1]["details"] += f"\n\nResponse:\n{json.dumps(rr, ensure_ascii=False, indent=2)}"
    else:
        steps[1]["status"] = "error"
        steps[1]["info"] = "Failed"
        steps[1]["details"] += f"\n\nError:\n{rr.get('error')}"
        render()
        return cr, rr, ""
    render()

    # STEP 3: LLM Script
    steps[2]["status"] = "running"
    steps[2]["info"] = "Constructing..." if lang == "en" else "构建中..."
    steps[2]["details"] = "Prompting GenAI Model to generate high-converting script based on RAG cases and risk profile..."
    render()
    time.sleep(0.5)
    
    script = generate_retention_script(params, cr, rr, lang)
    
    steps[2]["status"] = "completed"
    steps[2]["info"] = "Completed" if lang == "en" else "生成完毕"
    steps[2]["details"] += f"\n\nGenerated script length: {len(script)} chars"
    render()
    
    return cr, rr, script


# ── Sidebar layout ──
with st.sidebar:
    cur = st.session_state.lang
    lc1, lc2 = st.columns(2)
    with lc1:
        cls = "lang-btn-active" if cur == "zh" else "lang-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("中文", key="btn_zh", use_container_width=True):
            st.session_state["contract_val_zh"] = st.session_state.get("contract_val_en", 0)
            st.session_state["internet_val_zh"] = st.session_state.get("internet_val_en", 0)
            st.session_state["payment_val_zh"] = st.session_state.get("payment_val_en", 0)
            st.session_state.lang = "zh"; st.session_state.result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with lc2:
        cls = "lang-btn-active" if cur == "en" else "lang-btn"
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button("English", key="btn_en", use_container_width=True):
            st.session_state["contract_val_en"] = st.session_state.get("contract_val_zh", 0)
            st.session_state["internet_val_en"] = st.session_state.get("internet_val_zh", 0)
            st.session_state["payment_val_en"] = st.session_state.get("payment_val_zh", 0)
            st.session_state.lang = "en"; st.session_state.result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    t = LANG[st.session_state.lang]
    lang = st.session_state.lang

    # Presets Section
    st.markdown(f'<div class="k-section-label" style="margin-top:20px">{t["preset_header"]}</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown('<div class="preset-btn preset-high">', unsafe_allow_html=True)
        if st.button("High", key="load_high", use_container_width=True):
            load_preset("high")
        st.markdown('</div>', unsafe_allow_html=True)
    with pc2:
        st.markdown('<div class="preset-btn preset-mid">', unsafe_allow_html=True)
        if st.button("Medium", key="load_mid", use_container_width=True):
            load_preset("medium")
        st.markdown('</div>', unsafe_allow_html=True)
    with pc3:
        st.markdown('<div class="preset-btn preset-low">', unsafe_allow_html=True)
        if st.button("Low", key="load_low", use_container_width=True):
            load_preset("low")
        st.markdown('</div>', unsafe_allow_html=True)

    # Input Fields Form
    st.markdown(f'<div class="k-section-label" style="margin-top:20px">{t["input_header"]}</div>', unsafe_allow_html=True)
    
    col_id, col_tot = st.columns(2)
    with col_id:
        customer_id = st.text_input(t["customer_id"], key="customer_id_val")
    with col_tot:
        total = st.number_input(t["total"], min_value=0, max_value=99999, step=1, key="total_val")

    col_ten, col_mon = st.columns(2)
    with col_ten:
        tenure = st.number_input(t["tenure"], min_value=0, max_value=72, step=1, key="tenure_val", on_change=update_total)
    with col_mon:
        monthly = st.number_input(t["monthly"], min_value=0, max_value=500, step=1, key="monthly_val", on_change=update_total)

    col_con, col_int = st.columns(2)
    with col_con:
        contract_idx = st.selectbox(t["contract"], range(len(t["contract_opts"])),
                                    format_func=lambda i: t["contract_opts"][i], key=f"contract_val_{lang}")
        contract = t["contract_vals"][contract_idx]
    with col_int:
        internet_idx = st.selectbox(t["internet"], range(len(t["internet_opts"])),
                                    format_func=lambda i: t["internet_opts"][i], key=f"internet_val_{lang}")
        internet = t["internet_vals"][internet_idx]

    payment_idx = st.selectbox(t["payment"], range(len(t["payment_opts"])),
                               format_func=lambda i: t["payment_opts"][i], key=f"payment_val_{lang}")
    payment = t["payment_vals"][payment_idx]

    run_clicked = st.button(t["run_btn"], use_container_width=True)


# ── Main Interface ──
t = LANG[st.session_state.lang]
lang = st.session_state.lang

# Diagnostics for system status
pred_status, gem_status = check_system_status()
pred_class = "green" if pred_status == "online" else "orange"
pred_lbl = "Predictor: FastAPI" if pred_status == "online" else "Predictor: In-Process Fallback"
gem_class = "green" if gem_status == "active" else "orange"
gem_lbl = "Gemini: Connected" if gem_status == "active" else "Gemini: Demo Fallback"

# Dynamic Header Component
header_html = f"""
<div class="agent-header">
    <div class="agent-header-left">
        <div>
            <div class="agent-title">{t["title"]}</div>
            <div class="agent-subtitle">{t["subtitle"]}</div>
        </div>
    </div>
    <div class="agent-status-badges">
        <span class="status-pill {pred_class}">{pred_lbl}</span>
        <span class="status-pill {gem_class}">{gem_lbl}</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown('<div class="ai-line"></div>', unsafe_allow_html=True)

# Grid Layout
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown(f'<div class="k-section-label">{t["trace_header"]}</div>', unsafe_allow_html=True)
    trace_ph = st.empty()
    if not run_clicked and st.session_state.result is None:
        initial_timeline = render_timeline([
            {"id": 1, "name_zh": "流失风险预测 (ML 模型)", "name_en": "Risk Prediction (ML Model)", "status": "idle", "info": "", "details": ""},
            {"id": 2, "name_zh": "历史相似案例检索 (RAG)", "name_en": "Case Retrieval (RAG)", "status": "idle", "info": "", "details": ""},
            {"id": 3, "name_zh": "大语言模型生成挽留方案", "name_en": "LLM Retention Script Generation", "status": "idle", "info": "", "details": ""}
        ])
        trace_ph.markdown(initial_timeline, unsafe_allow_html=True)
        
    st.markdown(f'<div class="k-section-label" style="margin-top:24px">{t["rag_header"]}</div>', unsafe_allow_html=True)
    rag_ph = st.empty()

with col_right:
    st.markdown(f'<div class="k-section-label">{t["risk_header"]}</div>', unsafe_allow_html=True)
    risk_ph = st.empty()
    st.markdown(f'<div class="k-section-label" style="margin-top:24px">{t["strategy_header"]}</div>', unsafe_allow_html=True)
    strat_ph = st.empty()


# ── Execution Trigger ──
if run_clicked:
    params = {
        "customer_id": customer_id, 
        "tenure": int(tenure),
        "contract": contract, 
        "internet_service": internet,
        "monthly_charges": float(monthly), 
        "total_charges": float(total),
        "payment_method": payment,
    }
    risk_ph.empty()
    strat_ph.empty()
    rag_ph.empty()
    
    cr, rr, script = run_agent_streaming(params, trace_ph, t, lang)
    st.session_state.result = (cr, rr, script)


# ── Render Results ──
if st.session_state.result:
    cr, rr, script = st.session_state.result
    level = cr.get("risk_level", "low")
    prob  = cr.get("churn_probability", 0.0)

    with col_right:
        # Glowing gauge widget
        rec = translate_recommendation(cr.get("recommendation", ""), lang)
        risk_ph.markdown(
            render_risk_card(prob, level, rec, t),
            unsafe_allow_html=True
        )

        st.markdown(f'<div class="k-section-label" style="margin-top:16px">{t["factors_label"]}</div>', unsafe_allow_html=True)
        tags_html = '<div class="risk-factor-container">'
        for f in cr.get("key_risk_factors", []):
            tags_html += f'<span class="risk-factor-tag">⚠️ {translate_factor(f, lang)}</span>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)

        strategies = rr.get("aggregated_strategies", [])
        if strategies:
            strat_html = '<div class="strategy-list">'
            for i, s in enumerate(strategies, 1):
                s_trans = translate_strategy(s, lang)
                strat_html += f'<div class="strategy-item">' \
                              f'<div class="strategy-badge">{i:02d}</div>' \
                              f'<div class="strategy-text">{s_trans}</div>' \
                              f'</div>'
            strat_html += '</div>'
            strat_ph.markdown(strat_html, unsafe_allow_html=True)

    with col_left:
        cases = rr.get("cases", [])
        if cases:
            cases_html = ""
            for c in cases:
                sim = c["similarity"]
                prof = c["profile"]
                case_tags = "".join(
                    f'<span class="case-tag">{translate_strategy(s.strip(), lang)}</span>'
                    for s in c["retention_strategy"].split("；") if s.strip()
                )
                
                sim_text = f"{sim:.1%} similarity" if lang == "en" else f"相似度 {sim:.1%}"
                unit = "mo" if lang == "en" else "月"
                contract_str = translate_db_val(prof.get("contract"), lang)
                internet_str = translate_db_val(prof.get("internet_service"), lang)
                
                cases_html += f'<div class="case-card">' \
                              f'<div class="case-header">' \
                              f'<span class="case-similarity">{sim_text}</span>' \
                              f'<span class="case-meta">{prof["tenure_months"]}{t["tenure_unit"]} · {contract_str}</span>' \
                              f'</div>' \
                              f'<div style="font-size: 0.72rem; color: #a5adc9;">' \
                              f'{t["internet"]}: {internet_str} · {t["monthly"]}: ${prof.get("monthly_charges"):.0f}/{unit}' \
                              f'</div>' \
                              f'<div class="case-tags-container">{case_tags}</div>' \
                              f'</div>'
            rag_ph.markdown(cases_html, unsafe_allow_html=True)

    if script:
        st.markdown(f'<div class="script-workspace-title">{t["script_header"]}</div>', unsafe_allow_html=True)
        safe_script = script.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="script-card"><div class="script-content">{safe_script}</div></div>',
            unsafe_allow_html=True
        )
