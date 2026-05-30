import streamlit as st
import pandas as pd
import numpy as np
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
        color: #1e3a5f !important;
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
    .prediction-sub { font-size: 12px; font-weight: 600; color: #16a34a; }
    .prediction-sub-red { font-size: 12px; font-weight: 600; color: #dc2626; }
    .ai-box {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin-top: 16px;
        border-left: 5px solid #1d4ed8;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        color: #1e3a5f !important;
        font-size: 14px;
        line-height: 1.7;
    }
    .ai-box p, .ai-box li, .ai-box h1, .ai-box h2, .ai-box h3 {
        color: #1e3a5f !important;
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
    /* Fix markdown text color */
    .stMarkdown p { color: #1e3a5f !important; }
    .stMarkdown li { color: #1e3a5f !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

C1 = "#1e3a5f"
C2 = "#1d4ed8"
C3 = "#2563eb"

AXIS_STYLE = dict(
    tickfont=dict(size=12, color="#1e3a5f", family="Arial"),
    title_font=dict(size=13, color="#1e3a5f", family="Arial"),
    showgrid=True, gridcolor="#e2e8f0"
)
XAXIS_STYLE = dict(
    tickfont=dict(size=11, color="#1e3a5f", family="Arial"),
    title_font=dict(size=13, color="#1e3a5f", family="Arial"),
    showgrid=False, tickangle=-45
)
FONT  = dict(family="Arial", size=12, color="#1e3a5f")
MARG  = dict(t=55, b=70, l=60, r=20)
LEGEND = dict(font=dict(size=12, color="#1e3a5f", family="Arial"),
              bgcolor="white", bordercolor="#e2e8f0", borderwidth=1)

def fmt(n):
    if abs(n) >= 1_000_000_000:
        return f"${n/1_000_000_000:.1f}B"
    elif abs(n) >= 1_000_000:
        return f"${n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:,.0f}"

def call_groq(prompt):
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
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
    df["month"]   = df["transaction_date"].dt.strftime("%Y-%m")
    df["quarter"] = df["transaction_date"].dt.to_period("Q").astype(str)
    df["year"]    = df["transaction_date"].dt.year
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

# ── Q1: Churn Risk by Segment ─────────────────────────────────────────────────
if question == "Q1: Churn Risk by Segment":
    st.markdown('<div class="section-title">Q1: Which Customer Segment is Most Likely to Churn?</div>', unsafe_allow_html=True)

    # Correct churn calculation: unique churned customers / unique total customers per segment
    total_per_seg   = df.groupby("customer_segment")["customer_id"].nunique().reset_index()
    total_per_seg.columns = ["customer_segment", "total"]
    churned_per_seg = df[df["churn_flag"]=="Yes"].groupby("customer_segment")["customer_id"].nunique().reset_index()
    churned_per_seg.columns = ["customer_segment", "churned"]
    seg_churn = total_per_seg.merge(churned_per_seg, on="customer_segment", how="left")
    seg_churn["churned"]    = seg_churn["churned"].fillna(0)
    seg_churn["churn_rate"] = (seg_churn["churned"] / seg_churn["total"] * 100).round(2)
    seg_churn = seg_churn.sort_values("churn_rate", ascending=False)
    highest_risk = seg_churn.iloc[0]["customer_segment"]
    highest_rate = seg_churn.iloc[0]["churn_rate"]

    cols = st.columns(len(seg_churn))
    for i, (col, row) in enumerate(zip(cols, seg_churn.itertuples())):
        color = "prediction-sub-red" if i == 0 else "prediction-sub"
        badge = "⚠ Highest Risk" if i == 0 else "✓ Lower Risk"
        with col:
            st.markdown(f"""<div class="prediction-card">
                <div class="prediction-label">{row.customer_segment}</div>
                <div class="prediction-value">{row.churn_rate:.1f}%</div>
                <div class="{color}">{badge}</div>
            </div>""", unsafe_allow_html=True)

    fig = px.bar(seg_churn, x="customer_segment", y="churn_rate",
                 color="customer_segment",
                 color_discrete_sequence=[C1, C2, C3],
                 text=[f"{v:.1f}%" for v in seg_churn["churn_rate"]],
                 title="Churn Rate by Customer Segment")
    fig.update_traces(textposition="outside",
                      textfont=dict(size=13, color="#1e3a5f", family="Arial Black"),
                      marker_line_width=0)
    fig.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                      title_font=dict(size=15, color="#1e3a5f", family="Arial Black"),
                      xaxis=XAXIS_STYLE,
                      yaxis=dict(**AXIS_STYLE, ticksuffix="%"),
                      margin=MARG, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this churn prediction in business language: "
                f"The {highest_risk} segment has the highest churn rate at {highest_rate:.1f}%. "
                f"All segments: {seg_churn[['customer_segment','churn_rate']].to_string(index=False)}. "
                f"Provide: 1. What this means for the business 2. Why this segment may be churning "
                f"3. Specific retention actions the sales team should take. Be concise and actionable."
            )
            result = call_groq(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)

# ── Q2: Revenue Decline by Region ─────────────────────────────────────────────
elif question == "Q2: Revenue Decline by Region":
    st.markdown('<div class="section-title">Q2: Which Region is Expected to Experience Revenue Decline?</div>', unsafe_allow_html=True)

    region_quarterly = df.groupby(["region","year"])["revenue"].sum().reset_index()
    region_quarterly = region_quarterly.sort_values("year")

    quarters = sorted(df["quarter"].unique())
    region_q = df.groupby(["region","quarter"])["revenue"].sum().reset_index()

    if len(quarters) >= 2:
        last_q = quarters[-1]
        prev_q = quarters[-2]
        last   = region_q[region_q["quarter"]==last_q].set_index("region")["revenue"]
        prev   = region_q[region_q["quarter"]==prev_q].set_index("region")["revenue"]
        trend  = ((last - prev) / prev * 100).reset_index()
        trend.columns = ["region","growth_pct"]
        trend  = trend.sort_values("growth_pct")
        at_risk = trend[trend["growth_pct"] < 0]["region"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(trend, x="region", y="growth_pct",
                      color="growth_pct",
                      color_continuous_scale=["#dc2626","#93c5fd","#1d4ed8"],
                      text=[f"{v:.1f}%" for v in trend["growth_pct"]],
                      title="Revenue Growth % by Region (Last Quarter)")
        fig1.update_traces(textposition="outside",
                           textfont=dict(size=13, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=XAXIS_STYLE,
                           yaxis=dict(**AXIS_STYLE, ticksuffix="%"),
                           margin=MARG, height=400, showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(region_quarterly, x="year", y="revenue", color="region",
                       color_discrete_sequence=[C1, C2, C3, "#0891b2"],
                       markers=True,
                       title="Annual Revenue Trend by Region")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=12, color="#1e3a5f"),
                                      showgrid=False, tickmode="linear", dtick=1),
                           yaxis=dict(**AXIS_STYLE, tickformat="$,.0f"),
                           margin=MARG, height=400, legend=LEGEND)
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this regional revenue prediction: "
                f"Regional growth rates: {trend.to_string(index=False)}. "
                f"Regions at risk: {at_risk if at_risk else 'None'}. "
                f"Provide: 1. Which regions leadership should focus on 2. Likely causes "
                f"3. Specific actions to reverse the trend. Be concise and actionable."
            )
            result = call_groq(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)

# ── Q3: Most Profitable Channel ───────────────────────────────────────────────
elif question == "Q3: Most Profitable Channel":
    st.markdown('<div class="section-title">Q3: Which Channel Will Generate the Highest Profit Next 3 Months?</div>', unsafe_allow_html=True)

    channel_monthly = df.groupby(["channel","month"])["profit"].sum().reset_index()
    channel_monthly = channel_monthly.sort_values("month")

    channel_forecast = []
    for channel in channel_monthly["channel"].unique():
        ch_data = channel_monthly[channel_monthly["channel"]==channel]["profit"].values
        if len(ch_data) >= 3:
            avg_growth  = np.mean(np.diff(ch_data[-6:])) if len(ch_data) >= 6 else 0
            last_val    = ch_data[-1]
            forecast    = [last_val + avg_growth*(i+1) for i in range(3)]
            channel_forecast.append({
                "channel": channel,
                "avg_monthly_profit": np.mean(ch_data),
                "forecasted_3m_profit": sum(forecast),
                "trend": "📈 Growing" if avg_growth > 0 else "📉 Declining"
            })

    forecast_df  = pd.DataFrame(channel_forecast).sort_values("forecasted_3m_profit", ascending=False)
    best_channel = forecast_df.iloc[0]["channel"]

    cols = st.columns(len(forecast_df))
    for i, (col, row) in enumerate(zip(cols, forecast_df.itertuples())):
        color = "prediction-sub" if "Growing" in row.trend else "prediction-sub-red"
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
                           textfont=dict(size=12, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=XAXIS_STYLE,
                           yaxis=dict(**AXIS_STYLE, tickformat="$,.0f"),
                           margin=MARG, height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        channel_yearly = df.groupby(["channel","year"])["profit"].sum().reset_index()
        fig2 = px.line(channel_yearly, x="year", y="profit", color="channel",
                       color_discrete_sequence=[C1, C2, C3],
                       markers=True, title="Annual Profit Trend by Channel")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=12, color="#1e3a5f"),
                                      showgrid=False, tickmode="linear", dtick=1),
                           yaxis=dict(**AXIS_STYLE, tickformat="$,.0f"),
                           margin=MARG, height=400, legend=LEGEND)
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this channel profitability forecast: "
                f"Forecasted 3-month profits: {forecast_df[['channel','forecasted_3m_profit','trend']].to_string(index=False)}. "
                f"Best channel: {best_channel}. "
                f"Provide: 1. Which channel to prioritize 2. Resource allocation recommendations "
                f"3. Actions to maximize profit. Be concise and actionable."
            )
            result = call_groq(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)

# ── Q4: Revenue Target Risk ───────────────────────────────────────────────────
elif question == "Q4: Revenue Target Risk":
    st.markdown('<div class="section-title">Q4: What is the Probability of Missing Revenue Targets?</div>', unsafe_allow_html=True)

    monthly_rev = df.groupby("month")[["actual_revenue","budgeted_revenue"]].sum().reset_index()
    monthly_rev["missed"]       = (monthly_rev["actual_revenue"] < monthly_rev["budgeted_revenue"]).astype(int)
    monthly_rev["variance_pct"] = ((monthly_rev["actual_revenue"] - monthly_rev["budgeted_revenue"])
                                    / monthly_rev["budgeted_revenue"] * 100).round(2)
    monthly_rev = monthly_rev.sort_values("month")

    miss_rate     = monthly_rev["missed"].mean() * 100
    avg_variance  = monthly_rev["variance_pct"].mean()
    months_missed = int(monthly_rev["missed"].sum())
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
            <div class="prediction-value">{months_missed}/{total_months}</div>
            <div class="prediction-sub-red">Historical Record</div>
        </div>""", unsafe_allow_html=True)

    # Use yearly aggregation for cleaner x-axis
    yearly_rev = df.groupby("year")[["actual_revenue","budgeted_revenue"]].sum().reset_index()
    yearly_rev["variance_pct"] = ((yearly_rev["actual_revenue"] - yearly_rev["budgeted_revenue"])
                                   / yearly_rev["budgeted_revenue"] * 100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_rev["year"], y=yearly_rev["variance_pct"],
        marker_color=[C2 if v >= 0 else "#dc2626" for v in yearly_rev["variance_pct"]],
        text=[f"{v:.1f}%" for v in yearly_rev["variance_pct"]],
        textposition="outside",
        textfont=dict(size=13, color="#1e3a5f", family="Arial Black")
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=2)
    fig.update_layout(
        title=dict(text="Annual Revenue Variance % (Actual vs Budget)",
                   font=dict(size=15, color="#1e3a5f", family="Arial Black")),
        font=FONT, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=12, color="#1e3a5f"), showgrid=False,
                   tickmode="linear", dtick=1),
        yaxis=dict(**AXIS_STYLE, ticksuffix="%"),
        margin=MARG, height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            prompt = (
                f"You are a senior financial analyst. Explain this revenue target risk: "
                f"Historical miss rate: {miss_rate:.1f}%, Average variance: {avg_variance:.1f}%, "
                f"Months missed: {months_missed} out of {total_months}. "
                f"Provide: 1. Risk assessment 2. Key factors driving misses "
                f"3. Recommendations to improve target achievement. Be concise and actionable."
            )
            result = call_groq(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)

# ── Q5: Category Profitability Risk ───────────────────────────────────────────
elif question == "Q5: Category Profitability Risk":
    st.markdown('<div class="section-title">Q5: Which Product Category is Most at Risk of Declining Profitability?</div>', unsafe_allow_html=True)

    cat_yearly = df.groupby(["category","year"])["profit"].sum().reset_index()
    cat_yearly = cat_yearly.sort_values("year")

    cat_risk = []
    for cat in cat_yearly["category"].unique():
        cat_data = cat_yearly[cat_yearly["category"]==cat]["profit"].values
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
                      color_continuous_scale=["#dc2626","#93c5fd","#1d4ed8"],
                      text=[f"{v:+.1f}%" for v in risk_df["trend_pct"]],
                      title="Profit Trend % by Category (Latest Year)")
        fig1.update_traces(textposition="outside",
                           textfont=dict(size=13, color="#1e3a5f", family="Arial Black"),
                           marker_line_width=0)
        fig1.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=XAXIS_STYLE,
                           yaxis=dict(**AXIS_STYLE, ticksuffix="%"),
                           margin=MARG, height=400, showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.line(cat_yearly, x="year", y="profit", color="category",
                       color_discrete_sequence=[C1, C2, C3],
                       markers=True, title="Annual Profit Trend by Category")
        fig2.update_layout(font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                           title_font=dict(size=14, color="#1e3a5f", family="Arial Black"),
                           xaxis=dict(tickfont=dict(size=12, color="#1e3a5f"),
                                      showgrid=False, tickmode="linear", dtick=1),
                           yaxis=dict(**AXIS_STYLE, tickformat="$,.0f"),
                           margin=MARG, height=400, legend=LEGEND)
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("🤖 Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            prompt = (
                f"You are a senior data analyst. Explain this product category profitability risk: "
                f"Category trends: {risk_df[['category','trend_pct','risk']].to_string(index=False)}. "
                f"Highest risk: {highest_risk_cat}. "
                f"Provide: 1. Which categories need attention 2. Likely causes "
                f"3. Actions to protect or improve profitability. Be concise and actionable."
            )
            result = call_groq(prompt)
            st.markdown(f'<div class="ai-box">{result}</div>', unsafe_allow_html=True)

