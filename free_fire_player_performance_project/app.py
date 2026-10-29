from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Battle Royale Performance Predictor", page_icon="🎮", layout="wide")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "battle_royale_synthetic_matches.csv"

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    # fallback synthetic data generation if CSV isn't present
    rng = np.random.default_rng(42)
    n = 1800
    kills = np.clip(rng.poisson(3.2, n), 0, 18)
    assists = np.clip(rng.poisson(1.4, n), 0, 10)
    damage = np.clip(rng.gamma(2.2, 115, n) + kills*18 + assists*8, 0, 1800).round().astype(int)
    survival = np.clip(rng.normal(12 + kills*.45 + assists*.12, 5, n), 1, 28).round(1)
    placement = np.clip(np.round(11 - kills*.45 - survival*.16 + rng.normal(0, 3, n)), 1, 50).astype(int)
    accuracy = np.clip(rng.normal(25 + kills*1.2 + damage/100*1.5, 9, n), 5, 85).round(1)
    headshots = np.clip(np.round(kills * rng.uniform(.05, .55, n)), 0, kills).astype(int)
    revives = np.clip(rng.poisson(.8, n), 0, 8)
    duration = np.clip(survival + rng.uniform(1, 8, n), survival, 35).round(1)
    score = kills*7 + assists*3 + damage*.035 + survival*1.2 + accuracy*.1 + headshots*1.5 + revives + (51-placement)*.65 + rng.normal(0,8,n)
    q1, q2 = np.quantile(score, [1/3, 2/3])
    tier = np.where(score <= q1, "Low", np.where(score <= q2, "Medium", "High"))
    return pd.DataFrame({
        "kills": kills, "assists": assists, "damage": damage,
        "survival_minutes": survival, "placement": placement,
        "accuracy_percent": accuracy, "headshots": headshots,
        "revives": revives, "match_duration_minutes": duration,
        "performance_tier": tier, "demo_performance_score": score
    })

@st.cache_resource
def train_model(data):
    features = ["kills", "assists", "damage", "survival_minutes", "placement",
                "accuracy_percent", "headshots", "revives", "match_duration_minutes"]
    X = data[features]
    y = data["performance_tier"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, features, X_test, y_test, pred

data = load_data()
model, features, X_test, y_test, pred = train_model(data)

st.title("🎮 Battle Royale Player Performance Predictor")
st.caption("MCA Data Science Project | Machine Learning + Interactive Analytics")
st.warning("Demo mode: the bundled dataset is synthetic, not official Free Fire data. Predictions demonstrate the workflow and should not be interpreted as validated real-world performance.")
tab1, tab2, tab3, tab4 = st.tabs(["Predict Player", "Dataset Explorer", "Model Evaluation", "About Project"])

with tab1:
    st.subheader("Enter match statistics")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_kills = st.slider("Kills", 0, 20, 4)
            in_assists = st.slider("Assists", 0, 12, 2)
            in_damage = st.slider("Damage", 0, 1800, 450, step=25)
        with c2:
            in_survival = st.slider("Survival time (minutes)", 1.0, 30.0, 15.0, step=0.5)
            in_placement = st.slider("Final placement (1 = best)", 1, 50, 8)
            in_accuracy = st.slider("Accuracy (%)", 0.0, 100.0, 35.0, step=1.0)
        with c3:
            in_headshots = st.slider("Headshots", 0, 20, 2)
            in_revives = st.slider("Revives", 0, 10, 1)
            in_duration = st.slider("Match duration (minutes)", 1.0, 35.0, 20.0, step=0.5)
        submitted = st.form_submit_button("Predict performance", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "kills": in_kills, "assists": in_assists, "damage": in_damage,
            "survival_minutes": in_survival, "placement": in_placement,
            "accuracy_percent": in_accuracy, "headshots": min(in_headshots, in_kills),
            "revives": in_revives, "match_duration_minutes": max(in_duration, in_survival)
        }], columns=features)
        label = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        classes = model.classes_
        conf = float(probabilities[list(classes).index(label)])
        color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(label, "🎯")
        st.markdown(f"## {color} Predicted performance: **{label}**")
        st.metric("Model confidence (class probability)", f"{conf:.1%}")
        st.caption("This probability is model output, not a guarantee that the prediction is correct.")
        prob_df = pd.DataFrame({"Performance tier": classes, "Probability": probabilities}).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Performance tier"))

with tab2:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Match records", f"{len(data):,}")
    b.metric("Input features", len(features))
    c.metric("Performance categories", data["performance_tier"].nunique())
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download demo dataset (CSV)", data.to_csv(index=False).encode("utf-8"),
                       file_name="battle_royale_synthetic_matches.csv", mime="text/csv")
    st.subheader("Performance category distribution")
    counts = data["performance_tier"].value_counts().reindex(["Low", "Medium", "High"])
    st.bar_chart(counts)
    st.subheader("Kills vs damage")
    st.scatter_chart(data, x="kills", y="damage", color="performance_tier")

with tab3:
    st.subheader("Hold-out test-set evaluation")
    acc = accuracy_score(y_test, pred)
    st.metric("Accuracy on synthetic test split", f"{acc:.1%}")
    st.caption("This score measures performance on a held-out split of the synthetic dataset only.")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    cm = confusion_matrix(y_test, pred, labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"], ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    st.subheader("Feature importance")
    st.bar_chart(importance)

with tab4:
    st.subheader("Project summary")
    st.markdown("""
**Objective:** demonstrate how match statistics can be used to classify a player's performance tier.

**Algorithm:** Random Forest Classifier.

**Workflow:** synthetic dataset → preprocessing / feature selection → train-test split → model training → evaluation → Streamlit prediction interface.

**Input features:** kills, assists, damage, survival time, final placement, accuracy, headshots, revives, and match duration.

**Target label:** Low, Medium, or High. In this demo, labels are derived from a transparent composite score. Replace this target with a meaningful label or outcome from a real dataset for research-grade results.

**Limitations:** synthetic data is not representative of all real players, modes, maps, devices, or matchmaking conditions. The demo is educational and is not affiliated with Garena or Free Fire.
""")
