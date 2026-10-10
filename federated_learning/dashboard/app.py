import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import os

st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🌐",
    layout="wide"
)

AGGREGATOR = os.getenv("AGGREGATOR_URL", "http://aggregator:8000")
SITES = {
    "Site A (World)":    os.getenv("SITE_A_URL", "http://site_a:8000"),
    "Site B (Sports)":   os.getenv("SITE_B_URL", "http://site_b:8000"),
    "Site C (Business)": os.getenv("SITE_C_URL", "http://site_c:8000"),
    "Site D (Sci/Tech)": os.getenv("SITE_D_URL", "http://site_d:8000"),
}

st.title("🌐 Federated Learning Dashboard")
st.caption("Non-IID 4-Site BERT Training | FedAvg | Tsegai Yhdego PhD")
st.divider()

# Sidebar
st.sidebar.header("Training Control")
num_rounds = st.sidebar.slider("Rounds", 1, 10, 3)
epochs     = st.sidebar.slider("Epochs per round", 1, 3, 1)
lr         = st.sidebar.select_slider(
    "Learning rate",
    options=[1e-5, 2e-5, 3e-5, 5e-5],
    value=2e-5
)

if st.sidebar.button("▶ Run One Round", type="primary"):
    with st.spinner("Running round..."):
        try:
            r = requests.post(
                f"{AGGREGATOR}/round",
                json={"epochs": epochs, "lr": lr},
                timeout=1200
            )
            if r.status_code == 200:
                res = r.json()
                st.sidebar.success(
                    f"Round {res['round_num']} done!\n"
                    f"Accuracy: {res['global_accuracy']:.4f}"
                )
            else:
                st.sidebar.error(f"Error: {r.text[:200]}")
        except Exception as e:
            st.sidebar.error(str(e))
    st.rerun()

if st.sidebar.button("⏩ Run Multiple Rounds"):
    with st.spinner(f"Running {num_rounds} rounds..."):
        try:
            r = requests.post(
                f"{AGGREGATOR}/train",
                json={"num_rounds": num_rounds, "epochs": epochs, "lr": lr},
                timeout=7200
            )
            if r.status_code == 200:
                res = r.json()
                st.sidebar.success(
                    f"Done! Best: {res['best_accuracy']:.4f} "
                    f"(Round {res['best_round']})"
                )
        except Exception as e:
            st.sidebar.error(str(e))
    st.rerun()

if st.sidebar.button("🔄 Refresh"):
    st.rerun()

st.sidebar.divider()
st.sidebar.checkbox("Auto-refresh (30s)", key="auto_refresh")

# Aggregator status
try:
    agg = requests.get(f"{AGGREGATOR}/", timeout=5).json()
    hist = requests.get(f"{AGGREGATOR}/history", timeout=5).json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Round", agg["current_round"])
    col2.metric("Total Rounds Done", hist["total_rounds"])
    col3.metric(
        "Best Accuracy",
        f"{hist['best_accuracy']:.4f}" if hist["total_rounds"] > 0 else "N/A"
    )
    col4.metric("Status", "🟡 Training" if agg["is_training"] else "🟢 Ready")

except Exception as e:
    st.error(f"Cannot reach aggregator: {e}")
    st.stop()

st.divider()

# Site status
st.subheader("📡 Site Status")
cols = st.columns(4)
for i, (name, url) in enumerate(SITES.items()):
    with cols[i]:
        try:
            s = requests.get(f"{url}/", timeout=3).json()
            st.metric(name, f"{s.get('last_accuracy', 'N/A')}")
            st.caption(f"Round {s.get('round', 0)} | {s.get('device', 'cpu')}")
            if s.get("is_training"):
                st.warning("Training...")
            else:
                st.success("Ready")
        except:
            st.error("Offline")

st.divider()

# Training history
if hist["total_rounds"] > 0:
    st.subheader("📈 Training Progress")
    history_data = hist["history"]

    col_l, col_r = st.columns(2)

    with col_l:
        rounds = [r["round_num"] for r in history_data]
        accs   = [r["global_accuracy"] for r in history_data]
        imps   = [r["improvement"] for r in history_data]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=accs,
            mode="lines+markers+text",
            name="Global Accuracy",
            line=dict(color="#2196F3", width=3),
            marker=dict(size=10),
            text=[f"{a:.3f}" for a in accs],
            textposition="top center"
        ))
        fig.add_hline(y=0.333, line_dash="dash", line_color="red",
                     annotation_text="Single round baseline")
        fig.add_hline(y=0.843, line_dash="dash", line_color="green",
                     annotation_text="Centralized baseline")
        fig.update_layout(
            title="Convergence Curve",
            xaxis_title="Round",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        latest = history_data[-1]
        per_class = latest.get("per_class_accuracy", {})
        if per_class:
            fig2 = px.bar(
                x=list(per_class.keys()),
                y=list(per_class.values()),
                color=list(per_class.keys()),
                title=f"Per-Class Accuracy (Round {latest['round_num']})",
                color_discrete_sequence=["#636EFA","#EF553B","#00CC96","#AB63FA"]
            )
            fig2.update_layout(
                showlegend=False,
                yaxis_range=[0, 1],
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Results table
    st.subheader("📋 Round Summary")
    rows = []
    for r in history_data:
        rows.append({
            "Round":           r["round_num"],
            "Global Accuracy": f"{r['global_accuracy']:.4f}",
            "Improvement":     f"{r['improvement']:+.4f}",
            "Sites":           r.get("sites_responded", "?"),
            "Time (s)":        f"{r['round_time']:.1f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Site accuracy table per round
    if history_data[-1].get("site_results"):
        st.subheader("🏢 Site Local Accuracies")
        site_rows = []
        for r in history_data:
            for s in r.get("site_results", []):
                site_rows.append({
                    "Round": r["round_num"],
                    "Site": s["site"],
                    "Local Accuracy": f"{s['local_accuracy']:.4f}",
                    "Samples": s["num_samples"],
                    "Time (s)": f"{s['training_time']:.1f}"
                })
        st.dataframe(
            pd.DataFrame(site_rows),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No training rounds yet. Use the sidebar to start training.")

st.divider()
st.caption("Federated Learning Dashboard | DistilBERT | FedAvg | MLflow | "
           "Built by Tsegai Yhdego PhD — R-SEAT Center FAMU-FSU")

if st.session_state.get("auto_refresh"):
    time.sleep(30)
    st.rerun()
