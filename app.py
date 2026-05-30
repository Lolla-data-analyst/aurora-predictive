import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Aurora Predictive Analytics", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    .header-banner {
        background: linear-gradient(120deg, #1e3a5f 0%, #1d4ed8 100%);
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #1e3a5f;
        border-left: 4px solid #1d4ed8;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border-left: 5px solid #1d4ed8;
        margin-bottom: 16px;
    }
    .prediction-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-value {
        font-size: 24px;
        font-weight: 800;
        color: #1e3a5f;
        margin: 6px 0;
    }
    .prediction-sub {
        font-size: 12px;
        font-weight: 600;
        color: #16a34a;
    }
    .prediction-sub-red {
        font-size: 12px;
        font-weight: 600;
        color: #dc2626;
    }
    [data-testid="stSidebar"] { background-color: #1e3a5f; }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton > button {
        background-color: #1d4ed8;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

C1 = "#1e3a5f"
C2 = "#1d4ed8"
C3 = "#93c5fd"

def fmt(n):
    if abs(n) >= 1_000_000_000:
        return f"${n/1_000_000_000:.1f}B"
    elif abs(n) >= 1_000_000:
        return f"${n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:,.0f}"

def call_groq(prompt, api_key):
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile",
              "messages": [{"role": "user", "content": prompt}]}
    ).json()
    if "choices" in res:
        return res["choices"][0]["message"]["content"]
    return "Could not generate explanation."

@st.cache_data
def load_data():
    df = pd.read_excel("aurora_full_dataset.xlsx")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.strftime("%Y-%m")
    df["quarter"] = df["transaction_date"].dt.to_period("Q").astype(str)
    return df

df = load_data()

# Sidebar
st.sidebar.markdown("## 🔮 Aurora Predictive")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
question = st.sidebar.radio("Select Prediction", [
    "Q1: Churn Risk by Segment",
    "Q2: Revenue Decline by Region",
    "Q3: Most Profitable Channel",
    "Q4: Revenue Target Risk",
    "Q5: Category Profitability Risk"
])
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Groq AI")

# Header
st.markdown("""
<div class="header-banner">
    <h1 style="margin:0;font-size:24px;font-weight:800;color:white;">🔮 Aurora Predictive Sales & Customer Analytics</h1>
    <p style="margin:5px 0 0 0;font-size:13px;color:white;opacity:0.85;">AI-Powered Predictions & Business Recommendations</p>
</div>""", unsafe_allow_html=True)

FONT  = dict(family="Arial", size=12, color="#1e3a5f")
MARG  = dict(t=50, b=60, l=50, r=20)

# ── Q1: Churn Risk by Segment ─────────────────────────────────────────────────
if question == "Q1: Churn Risk by Segment":
    st.markdown('<div class="section-title">Q1: Which Customer Segment is Most Likely to Churn?</div>', unsafe_allow_html=True)

    seg_churn = df.groupby("customer_segment").agg(
        total=("customer_id", "nunique"),
        churned=("churn_flag", lambda x: (x == "Yes").sum())
    ).reset_index()
    seg_churn["churn_rate"] = (seg_churn["churned"] / seg_churn["total"] * 100).round(2)
    seg_churn = seg_churn.sort_values("churn_rate", ascending=False)
    highest_risk = seg_churn.iloc[0]["customer_segment"]
    highest_rate = seg_churn.iloc[0]["churn_rate"]

    c1, c2, c3 = st.columns(3)
    for i, (col, row) in enumerate(zip([c1, c2, c3], seg_churn.itertuples())):
        color = "prediction-sub-red" if i == 0 else "prediction-sub"
        with col:
            st.markdown(f"""<div class="prediction-card">
                <div class="prediction-label">{row.customer_segment}</div>
                <div class="prediction-value">{row.churn_rate:.1f}%</div>
                <div class="{color}">{'⚠ Highest Risk' if i == 0 else '✓ Lower Risk'}</div>
            </div>""", unsafe_allow_html=True)

    fig = px.bar(seg_churn, x="customer_segment", y="churn_rate",
                 color="customer_segment",
                 color_discrete_sequence=[C1, C2, C3],
                 text=[f"{v:.1f}%" for v in seg_churn["churn_rate"]],
                 title="Churn Rate by Customer Segment")
    fig.update_traces(textposition="outside",
                      textfont=dict(size=12, color="#1e3a5f", family="Arial Black"),
                      marker_line_width=0)
    fig.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                      title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                      xaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=False),
                      yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=True,
                                 gridcolor="#e2e8f0", ticksuffix="%"),
                      margin=MARG, height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating prediction explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this churn prediction result in business language: "
                f"The {highest_risk} segment has the highest churn rate at {highest_rate:.1f}%. "
                f"Churn rates by segment: {seg_churn[['customer_segment','churn_rate']].to_string(index=False)}. "
                f"Provide: 1. What this means for the business 2. Why this segment may be churning "
                f"3. Specific retention actions the sales team should take immediately. Be concise and actionable."
            )
            st.markdown(call_groq(prompt, st.secrets["GROQ_API_KEY"]))

# ── Q2: Revenue Decline by Region ─────────────────────────────────────────────
elif question == "Q2: Revenue Decline by Region":
    st.markdown('<div class="section-title">Q2: Which Region is Expected to Experience Revenue Decline?</div>', unsafe_allow_html=True)

    region_monthly = df.groupby(["region", "quarter"])["revenue"].sum().reset_index()
    quarters = sorted(region_monthly["quarter"].unique())
    if len(quarters) >= 2:
        last_q  = quarters[-1]
        prev_q  = quarters[-2]
        last    = region_monthly[region_monthly["quarter"] == last_q].set_index("region")["revenue"]
        prev    = region_monthly[region_monthly["quarter"] == prev_q].set_index("region")["revenue"]
        trend   = ((last - prev) / prev * 100).reset_index()
        trend.columns = ["region", "growth_pct"]
        trend   = trend.sort_values("growth_pct")
        at_risk = trend[trend["growth_pct"] < 0]

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(trend, x="region", y="growth_pct",
                      color="growth_pct",
                      color_continuous_scale=["#dc2626", "#93c5fd", "#1d4ed8"],
                      text=[f"{v:.1f}%" for v in trend["growth_pct"]],
                      title="Revenue Growth % by Region (Last Quarter)")
        fig1.update_traces(textposition="outside",
                           textfont=dict(size=12, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=False),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=True,
                                      gridcolor="#e2e8f0", ticksuffix="%"),
                           margin=MARG, height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        region_trend = df.groupby(["region", "quarter"])["revenue"].sum().reset_index()
        fig2 = px.line(region_trend, x="quarter", y="revenue", color="region",
                       color_discrete_sequence=[C1, C2, C3, "#0891b2"],
                       title="Revenue Trend by Region Over Time")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=10, color="#1e3a5f"),
                                      showgrid=False, tickangle=-45),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"),
                                      showgrid=True, gridcolor="#e2e8f0", tickformat="$,.0f"),
                           margin=MARG, height=380,
                           legend=dict(font=dict(size=11, color="#1e3a5f")))
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating prediction explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this regional revenue prediction: "
                f"Regional growth rates: {trend.to_string(index=False)}. "
                f"Regions at risk of decline: {at_risk['region'].tolist() if len(at_risk) > 0 else 'None'}. "
                f"Provide: 1. Which regions leadership should focus on 2. Likely causes of decline "
                f"3. Specific actions to reverse the trend. Be concise and actionable."
            )
            st.markdown(call_groq(prompt, st.secrets["GROQ_API_KEY"]))

# ── Q3: Most Profitable Channel ───────────────────────────────────────────────
elif question == "Q3: Most Profitable Channel":
    st.markdown('<div class="section-title">Q3: Which Channel Will Generate the Highest Profit Next 3 Months?</div>', unsafe_allow_html=True)

    channel_monthly = df.groupby(["channel", "month"])["profit"].sum().reset_index()
    channel_monthly = channel_monthly.sort_values("month")

    channel_forecast = []
    for channel in channel_monthly["channel"].unique():
        ch_data = channel_monthly[channel_monthly["channel"] == channel]["profit"].values
        if len(ch_data) >= 3:
            avg_growth = np.mean(np.diff(ch_data[-6:])) if len(ch_data) >= 6 else 0
            last_val   = ch_data[-1]
            forecast   = [last_val + avg_growth * (i+1) for i in range(3)]
            channel_forecast.append({
                "channel": channel,
                "avg_monthly_profit": np.mean(ch_data),
                "forecasted_3m_profit": sum(forecast),
                "trend": "📈 Growing" if avg_growth > 0 else "📉 Declining"
            })

    forecast_df = pd.DataFrame(channel_forecast).sort_values("forecasted_3m_profit", ascending=False)
    best_channel = forecast_df.iloc[0]["channel"]

    cols = st.columns(len(forecast_df))
    for i, (col, row) in enumerate(zip(cols, forecast_df.itertuples())):
        color = "prediction-sub" if i == 0 else "prediction-sub-red" if "Declining" in row.trend else "prediction-sub"
        with col:
            st.markdown(f"""<div class="prediction-card">
                <div class="prediction-label">{row.channel}</div>
                <div class="prediction-value">{fmt(row.forecasted_3m_profit)}</div>
                <div class="{color}">{row.trend} {'⭐ Best' if i == 0 else ''}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(forecast_df, x="channel", y="forecasted_3m_profit",
                      color="channel", color_discrete_sequence=[C1, C2, C3],
                      text=[fmt(v) for v in forecast_df["forecasted_3m_profit"]],
                      title="Forecasted 3-Month Profit by Channel")
        fig1.update_traces(textposition="outside",
                           textfont=dict(size=11, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=False),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=True,
                                      gridcolor="#e2e8f0", tickformat="$,.0f"),
                           margin=MARG, height=380, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(channel_monthly, x="month", y="profit", color="channel",
                       color_discrete_sequence=[C1, C2, C3],
                       title="Monthly Profit Trend by Channel")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=10, color="#1e3a5f"),
                                      showgrid=False, tickangle=-45, nticks=12),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"),
                                      showgrid=True, gridcolor="#e2e8f0", tickformat="$,.0f"),
                           margin=MARG, height=380,
                           legend=dict(font=dict(size=11, color="#1e3a5f")))
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating prediction explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this channel profitability forecast: "
                f"Forecasted 3-month profits: {forecast_df[['channel','forecasted_3m_profit','trend']].to_string(index=False)}. "
                f"Best channel: {best_channel}. "
                f"Provide: 1. Which channel to prioritize and why 2. Resource allocation recommendations "
                f"3. Actions to maximize profit in top channel. Be concise and actionable."
            )
            st.markdown(call_groq(prompt, st.secrets["GROQ_API_KEY"]))

# ── Q4: Revenue Target Risk ───────────────────────────────────────────────────
elif question == "Q4: Revenue Target Risk":
    st.markdown('<div class="section-title">Q4: What is the Probability of Missing Revenue Targets?</div>', unsafe_allow_html=True)

    monthly_rev = df.groupby("month")[["actual_revenue", "budgeted_revenue"]].sum().reset_index()
    monthly_rev["missed"] = (monthly_rev["actual_revenue"] < monthly_rev["budgeted_revenue"]).astype(int)
    monthly_rev["variance_pct"] = ((monthly_rev["actual_revenue"] - monthly_rev["budgeted_revenue"])
                                    / monthly_rev["budgeted_revenue"] * 100).round(2)

    miss_rate = monthly_rev["missed"].mean() * 100
    avg_variance = monthly_rev["variance_pct"].mean()
    months_missed = monthly_rev["missed"].sum()
    total_months  = len(monthly_rev)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="prediction-card">
            <div class="prediction-label">🎯 Miss Probability</div>
            <div class="prediction-value">{miss_rate:.1f}%</div>
            <div class="{'prediction-sub-red' if miss_rate > 40 else 'prediction-sub'}">
                {'⚠ High Risk' if miss_rate > 40 else '✓ Low Risk'}
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="prediction-card">
            <div class="prediction-label">📊 Avg Variance</div>
            <div class="prediction-value">{avg_variance:.1f}%</div>
            <div class="{'prediction-sub' if avg_variance > 0 else 'prediction-sub-red'}">
                {'▲ Over Budget' if avg_variance > 0 else '▼ Under Budget'}
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="prediction-card">
            <div class="prediction-label">📅 Months Missed</div>
            <div class="prediction-value">{int(months_missed)}/{total_months}</div>
            <div class="prediction-sub-red">Historical Record</div>
        </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_rev["month"], y=monthly_rev["variance_pct"],
                         marker_color=[C1 if v >= 0 else "#dc2626" for v in monthly_rev["variance_pct"]],
                         text=[f"{v:.1f}%" for v in monthly_rev["variance_pct"]],
                         textposition="outside",
                         textfont=dict(size=9, color="#1e3a5f", family="Arial Black")))
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=1.5)
    fig.update_layout(
        title=dict(text="Monthly Revenue Variance % (Actual vs Budget)",
                   font=dict(size=14, color="#1e3a5f", family="Arial Black")),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=10, color="#1e3a5f"), showgrid=False,
                   tickangle=-45, nticks=12),
        yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=True,
                   gridcolor="#e2e8f0", ticksuffix="%"),
        margin=MARG, height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating prediction explanation..."):
            prompt = (
                f"You are a senior financial analyst. Explain this revenue target risk assessment: "
                f"Historical miss rate: {miss_rate:.1f}%, Average variance: {avg_variance:.1f}%, "
                f"Months missed: {int(months_missed)} out of {total_months}. "
                f"Provide: 1. Assessment of revenue target risk 2. Key factors driving misses "
                f"3. Recommendations to improve target achievement. Be concise and actionable."
            )
            st.markdown(call_groq(prompt, st.secrets["GROQ_API_KEY"]))

# ── Q5: Category Profitability Risk ───────────────────────────────────────────
elif question == "Q5: Category Profitability Risk":
    st.markdown('<div class="section-title">Q5: Which Product Category is Most at Risk of Declining Profitability?</div>', unsafe_allow_html=True)

    cat_quarterly = df.groupby(["category", "quarter"])["profit"].sum().reset_index()
    cat_quarterly = cat_quarterly.sort_values("quarter")

    cat_risk = []
    for cat in cat_quarterly["category"].unique():
        cat_data = cat_quarterly[cat_quarterly["category"] == cat]["profit"].values
        if len(cat_data) >= 2:
            trend_val = cat_data[-1] - cat_data[-2]
            trend_pct = (trend_val / abs(cat_data[-2]) * 100) if cat_data[-2] != 0 else 0
            cat_risk.append({
                "category": cat,
                "latest_profit": cat_data[-1],
                "trend_pct": round(trend_pct, 2),
                "risk": "🔴 High Risk" if trend_pct < -5 else "🟡 Medium Risk" if trend_pct < 0 else "🟢 Low Risk"
            })

    risk_df = pd.DataFrame(cat_risk).sort_values("trend_pct")
    highest_risk_cat = risk_df.iloc[0]["category"]

    cols = st.columns(len(risk_df))
    for i, (col, row) in enumerate(zip(cols, risk_df.itertuples())):
        color = "prediction-sub-red" if "High" in row.risk else "prediction-sub"
        with col:
            st.markdown(f"""<div class="prediction-card">
                <div class="prediction-label">{row.category}</div>
                <div class="prediction-value">{row.trend_pct:+.1f}%</div>
                <div class="{color}">{row.risk}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(risk_df, x="category", y="trend_pct",
                      color="trend_pct",
                      color_continuous_scale=["#dc2626", "#93c5fd", "#1d4ed8"],
                      text=[f"{v:+.1f}%" for v in risk_df["trend_pct"]],
                      title="Profit Trend % by Category (Latest Quarter)")
        fig1.update_traces(textposition="outside",
                           textfont=dict(size=12, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=False),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"), showgrid=True,
                                      gridcolor="#e2e8f0", ticksuffix="%"),
                           margin=MARG, height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(cat_quarterly, x="quarter", y="profit", color="category",
                       color_discrete_sequence=[C1, C2, C3],
                       title="Profit Trend by Category Over Time")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=10, color="#1e3a5f"),
                                      showgrid=False, tickangle=-45),
                           yaxis=dict(tickfont=dict(size=11, color="#1e3a5f"),
                                      showgrid=True, gridcolor="#e2e8f0", tickformat="$,.0f"),
                           margin=MARG, height=380,
                           legend=dict(font=dict(size=11, color="#1e3a5f")))
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating prediction explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this product category profitability risk: "
                f"Category trends: {risk_df[['category','trend_pct','risk']].to_string(index=False)}. "
                f"Highest risk category: {highest_risk_cat}. "
                f"Provide: 1. Assessment of which categories need attention 2. Likely causes "
                f"3. Specific actions to protect or improve profitability. Be concise and actionable."
            )
            st.markdown(call_groq(prompt, st.secrets["GROQ_API_KEY"]))
