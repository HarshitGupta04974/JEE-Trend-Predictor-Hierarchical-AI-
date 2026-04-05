import re
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# sklearn / model imports — required for joblib unpickling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# SUBJECT EXPERT CLASS (Matches training script)
# =============================================================================
class SubjectExpert:
    def __init__(self, subject_name: str, model_type: str = "xgb"):
        self.subject_name  = subject_name
        self.model_type    = model_type
        self.vectorizer    = TfidfVectorizer(max_features=500, stop_words="english")
        self.nb_purity     = MultinomialNB()
        self.knn_template  = KNeighborsClassifier(metric="cosine")
        self.scaler        = StandardScaler()
        if model_type == "svm":
            base_clf = SVC(probability=True, kernel="linear", random_state=42)
        elif model_type == "rf":
            base_clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        else:
            base_clf = XGBClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=6, random_state=42, verbosity=0)
        self.clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

    def get_signals(self, df_input, features):
        X_num_scaled = self.scaler.transform(df_input[features])
        prob_main    = self.clf.predict_proba(X_num_scaled)[:, 1]
        X_text       = self.vectorizer.transform(df_input["Raw_Latex"])
        prob_knn     = self.knn_template.predict_proba(X_text)[:, 1]
        score_purity = np.max(self.nb_purity.predict_proba(X_text), axis=1)
        return prob_main, prob_knn, score_purity


FEATURES = ["Chapter_Code", "Year", "symbol_count",
            "Historical_Weight", "Is_Numerical", "years_since_last"]

SUBJECT_COLORS = {
    "Physics":     "#38bdf8",
    "Chemistry":   "#34d399",
    "Mathematics": "#f472b6",
}

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="JEE 2026 · AI Forecast",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #080c14;
}

/* ---- top hero strip ---- */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 300;
}

/* ---- stat cards ---- */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #0f1623;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #38bdf8);
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.stat-delta {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* ---- topic cards ---- */
.topic-card {
    background: #0f1623;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    position: relative;
    transition: border-color 0.2s;
}
.topic-card:hover { border-color: #334155; }
.topic-pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    margin-bottom: 0.45rem;
    background: var(--pill-bg);
    color: var(--pill-fg);
}
.topic-name {
    font-size: 0.92rem;
    font-weight: 500;
    color: #cbd5e1;
    margin-bottom: 0.45rem;
    line-height: 1.3;
}
.prob-bar-bg {
    background: #1e2a3a;
    border-radius: 4px;
    height: 4px;
    width: 100%;
}
.prob-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: var(--accent);
}
.prob-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    text-align: right;
    margin-top: 0.25rem;
}

/* ---- inference card ---- */
.inference-result {
    background: #0f1623;
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.infer-score {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.infer-label { color: #475569; font-size: 0.85rem; }

/* ---- disclaimer ---- */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #fbbf24;
    font-size: 0.8rem;
    margin-bottom: 1.2rem;
}

/* ---- section header ---- */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a3a;
}

/* ---- review card ---- */
.review-card {
    background: #0f1623;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}

/* ---- plotly overrides ---- */
.js-plotly-plot .plotly { background: transparent !important; }

/* ---- tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: #0f1623;
    border-radius: 10px;
    padding: 0.3rem;
    border: 1px solid #1e2a3a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 7px;
    font-size: 0.85rem;
    padding: 0.45rem 1rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e2a3a !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA & MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        experts     = joblib.load("trained_experts.pkl")
        manager     = joblib.load("meta_manager.pkl")
        meta_scaler = joblib.load("meta_scaler.pkl")
        le_chap     = joblib.load("label_encoder.pkl")
        return experts, manager, meta_scaler, le_chap, True
    except Exception as e:
        return None, None, None, None, False


@st.cache_data(show_spinner=False)
def load_forecast():
    try:
        df = pd.read_csv("final_forecast_2026.csv")
        return df, True
    except Exception:
        return pd.DataFrame(), False


@st.cache_data(show_spinner=False)
def load_historical():
    try:
        df = pd.read_csv("jee_full_dataset.csv")
        return df
    except Exception:
        try:
            df = pd.read_csv("final_forecast_2026.csv")
            return df
        except Exception:
            return pd.DataFrame()


experts, manager, meta_scaler, le_chap, models_loaded = load_models()
forecast_df, forecast_loaded = load_forecast()
historical_df = load_historical()


# =============================================================================
# DATABASE
# =============================================================================
def init_db():
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT, rating INTEGER, comment TEXT, date TEXT)""")
    conn.commit()
    conn.close()

def add_review(name, rating, comment):
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    conn.execute("INSERT INTO reviews (name,rating,comment,date) VALUES (?,?,?,?)",
                 (name, rating, comment, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_reviews():
    try:
        conn = sqlite3.connect("reviews.db", check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM reviews ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

init_db()


# =============================================================================
# PLOTLY THEME HELPER
# =============================================================================
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a0f1a",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
)
AXIS_STYLE = dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a")


# =============================================================================
# HERO
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">NTA Pattern Intelligence · 2026 Edition</div>
    <div class="hero-title">JEE Advanced Forecast</div>
    <div class="hero-sub">Hierarchical AI · 3-Expert Stack · 61% Precision on 2025 Validation</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> Probabilistic forecasts based on NTA historical patterns.
    Use as a supplementary strategy tool only. Full NCERT syllabus coverage remains essential.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# STAT CARDS
# =============================================================================
if not forecast_df.empty:
    n_topics   = len(forecast_df)
    mean_conf  = forecast_df["Final_Prob"].mean()
    top_conf   = forecast_df["Final_Prob"].max()
    n_chapters = forecast_df["Chapter"].nunique() if "Chapter" in forecast_df.columns else "—"
    hits       = int(forecast_df["Is_Hit"].sum()) if "Is_Hit" in forecast_df.columns else "—"
    precision  = f"{hits/n_topics:.0%}" if isinstance(hits, int) else "—"
else:
    n_topics, mean_conf, top_conf, n_chapters, precision = 90, 0, 0, "—", "—"

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card" style="--accent:#38bdf8">
    <div class="stat-label">Predicted Topics</div>
    <div class="stat-value">{n_topics}</div>
    <div class="stat-delta">30 per subject</div>
  </div>
  <div class="stat-card" style="--accent:#34d399">
    <div class="stat-label">Mean Confidence</div>
    <div class="stat-value">{mean_conf:.0%}</div>
    <div class="stat-delta">across all subjects</div>
  </div>
  <div class="stat-card" style="--accent:#f472b6">
    <div class="stat-label">Top Score</div>
    <div class="stat-value">{top_conf:.0%}</div>
    <div class="stat-delta">highest single topic</div>
  </div>
  <div class="stat-card" style="--accent:#a78bfa">
    <div class="stat-label">Validation Precision</div>
    <div class="stat-value">{precision}</div>
    <div class="stat-delta">on 2025 paper</div>
  </div>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    st.success("Models loaded — live inference available", icon="✅")
else:
    st.warning("Model files not found. Live inference disabled. Run training script first.", icon="⚠️")


# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "Prediction Matrix",
    "Live Inference",
    "Forecast Analytics",
    "Historical Trends",
    "Community",
])


# ---------------------------------------------------------------------------
# TAB 0 — PREDICTION MATRIX
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-header">Target Topic Matrix</div>', unsafe_allow_html=True)

    if forecast_df.empty:
        st.info("Run the training script to generate final_forecast_2026.csv")
    else:
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            selected_subs = st.multiselect(
                "Filter by subject",
                ["Physics", "Chemistry", "Mathematics"],
                default=["Physics", "Chemistry", "Mathematics"],
            )
        with fc2:
            min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, 0.05)
        with fc3:
            sort_by = st.selectbox("Sort by", ["Final_Prob ↓", "Subtopic A–Z", "Chapter A–Z"])

        view = forecast_df[
            forecast_df["Subject"].isin(selected_subs) &
            (forecast_df["Final_Prob"] >= min_prob)
        ].copy()

        if sort_by == "Final_Prob ↓":
            view = view.sort_values("Final_Prob", ascending=False)
        elif sort_by == "Subtopic A–Z":
            view = view.sort_values("Subtopic")
        else:
            view = view.sort_values("Chapter") if "Chapter" in view.columns else view

        st.caption(f"Showing {len(view)} topics")

        col_a, col_b, col_c = st.columns(3)
        cols = [col_a, col_b, col_c]

        for i, (_, row) in enumerate(view.iterrows()):
            subj    = row["Subject"]
            accent  = SUBJECT_COLORS.get(subj, "#94a3b8")
            prob    = row["Final_Prob"]
            bar_w   = int(prob * 100)
            chapter = row.get("Chapter", "")
            hit_tag = ""
            if "Is_Hit" in row and not pd.isna(row["Is_Hit"]):
                hit_tag = "✓ " if row["Is_Hit"] == 1 else ""

            with cols[i % 3]:
                st.markdown(f"""
                <div class="topic-card">
                  <span class="topic-pill" style="--pill-bg:rgba(255,255,255,0.05);--pill-fg:{accent}">
                    {subj.upper()}
                  </span>
                  <div class="topic-name">{hit_tag}{row['Subtopic']}</div>
                  {'<div style="color:#475569;font-size:0.75rem;margin-bottom:0.4rem">'+chapter+'</div>' if chapter else ''}
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{bar_w}%;--accent:{accent}"></div>
                  </div>
                  <div class="prob-label" style="--accent:{accent}">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.download_button(
            "Download forecast CSV",
            data=forecast_df.to_csv(index=False),
            file_name="jee_2026_forecast.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# TAB 1 — LIVE INFERENCE
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-header">Live Topic Inference</div>', unsafe_allow_html=True)

    inf_col1, inf_col2 = st.columns([1, 1], gap="large")

    with inf_col1:
        st.markdown("**Enter topic details**")
        u_topic   = st.text_input("Sub-topic name", placeholder="e.g. Rotational Dynamics")
        u_latex   = st.text_area("LaTeX representation (optional)",
                                  placeholder=r"e.g. \tau = I\alpha, \vec{L} = I\vec{\omega}",
                                  height=90)
        u_subj    = st.selectbox("Subject", ["Physics", "Chemistry", "Mathematics"])
        u_chapter = st.text_input("Chapter name", placeholder="e.g. rotational-mechanics")
        u_year_last = st.number_input("Last year this topic appeared", min_value=2010,
                                      max_value=2025, value=2023)
        u_numerical = st.toggle("Integer / Numerical type question?")
        u_symbols   = st.number_input("Approx. LaTeX command count (symbol density)",
                                      min_value=0, max_value=100, value=10)

        run_btn = st.button("Run Inference", type="primary", use_container_width=True)

    with inf_col2:
        if run_btn:
            if not models_loaded:
                st.error("Models not loaded. Run the training script first.")
            elif not u_topic:
                st.warning("Please enter a sub-topic name.")
            else:
                with st.spinner("Running through expert stack…"):
                    try:
                        known_chapters = set(le_chap.classes_)
                        chapter_code   = (
                            le_chap.transform([u_chapter])[0]
                            if u_chapter in known_chapters else 0
                        )
                        latex_text = u_latex if u_latex.strip() else (
                            " ".join(re.findall(r"\\[a-zA-Z]+", u_latex)) or u_topic
                        )
                        years_since = 2026 - int(u_year_last)

                        row_dict = {
                            "Raw_Latex"        : latex_text,
                            "Chapter_Code"     : chapter_code,
                            "Year"             : 2026,
                            "symbol_count"     : u_symbols,
                            "Historical_Weight": 10,
                            "Is_Numerical"     : int(u_numerical),
                            "years_since_last" : years_since,
                        }
                        row_df = pd.DataFrame([row_dict])

                        exp              = experts[u_subj]
                        p_m, p_k, s_p   = exp.get_signals(row_df, FEATURES)
                        meta_input       = meta_scaler.transform(
                            [[p_m[0], p_k[0], s_p[0], row_dict["Historical_Weight"]]]
                        )
                        final_prob = manager.predict_proba(meta_input)[0][1]

                        if final_prob >= 0.75:
                            score_color, verdict = "#34d399", "HIGH PRIORITY"
                        elif final_prob >= 0.50:
                            score_color, verdict = "#fbbf24", "MODERATE"
                        else:
                            score_color, verdict = "#f87171", "LOW PRIORITY"

                        st.markdown(f"""
                        <div class="inference-result">
                          <div class="infer-score" style="color:{score_color}">
                            {final_prob:.1%}
                          </div>
                          <div class="infer-label" style="color:{score_color};
                               font-family:'Space Mono',monospace;font-size:0.7rem;
                               letter-spacing:0.12em;margin-bottom:1.2rem">
                            {verdict}
                          </div>
                          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                                      gap:0.8rem;margin-top:1rem">
                            <div style="background:#0a0f1a;border-radius:8px;padding:0.7rem">
                              <div style="font-size:0.65rem;color:#475569;
                                          font-family:'Space Mono',monospace;
                                          letter-spacing:0.1em;margin-bottom:0.3rem">PRIMARY</div>
                              <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">{p_m[0]:.1%}</div>
                            </div>
                            <div style="background:#0a0f1a;border-radius:8px;padding:0.7rem">
                              <div style="font-size:0.65rem;color:#475569;
                                          font-family:'Space Mono',monospace;
                                          letter-spacing:0.1em;margin-bottom:0.3rem">KNN</div>
                              <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">{p_k[0]:.1%}</div>
                            </div>
                            <div style="background:#0a0f1a;border-radius:8px;padding:0.7rem">
                              <div style="font-size:0.65rem;color:#475569;
                                          font-family:'Space Mono',monospace;
                                          letter-spacing:0.1em;margin-bottom:0.3rem">PURITY</div>
                              <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0">{s_p[0]:.1%}</div>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=round(final_prob * 100, 1),
                            number={"suffix": "%", "font": {"size": 36, "color": score_color}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#1e2a3a"},
                                "bar":  {"color": score_color, "thickness": 0.25},
                                "bgcolor": "#0a0f1a",
                                "bordercolor": "#1e2a3a",
                                "steps": [
                                    {"range": [0, 50],  "color": "#0f1623"},
                                    {"range": [50, 75], "color": "#12202f"},
                                    {"range": [75, 100],"color": "#152535"},
                                ],
                                "threshold": {
                                    "line": {"color": score_color, "width": 2},
                                    "thickness": 0.75,
                                    "value": final_prob * 100,
                                },
                            },
                            title={"text": u_topic, "font": {"color": "#94a3b8", "size": 13}},
                        ))
                        fig_g.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=240,
                            margin=dict(l=20, r=20, t=30, b=10),
                            font=dict(color="#94a3b8"),
                        )
                        st.plotly_chart(fig_g, use_container_width=True)

                    except Exception as e:
                        st.error(f"Inference error: {e}")
        else:
            st.markdown("""
            <div style="background:#0f1623;border:1px solid #1e2a3a;border-radius:14px;
                        padding:2.5rem;text-align:center;color:#334155">
              <div style="font-size:2rem;margin-bottom:0.8rem">🔬</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.75rem;letter-spacing:0.1em">
                Fill in the form and click Run Inference
              </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 2 — FORECAST ANALYTICS
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-header">Forecast Analytics</div>', unsafe_allow_html=True)

    if forecast_df.empty:
        st.info("No forecast data found.")
    else:
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
                h = hex_color.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            fig1 = go.Figure()
            for subj, color in SUBJECT_COLORS.items():
                sub_data = forecast_df[forecast_df["Subject"] == subj]["Final_Prob"]
                if not sub_data.empty:
                    fig1.add_trace(go.Violin(
                        y=sub_data, name=subj, line_color=color,
                        fillcolor=hex_to_rgba(color, 0.15),
                        box_visible=True, meanline_visible=True, points="all",
                        pointpos=-1.5, marker=dict(size=3, color=color, opacity=0.5),
                    ))
            fig1.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Probability distribution by subject", font=dict(size=13)),
                showlegend=False, height=320,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with r1c2:
            top15 = forecast_df.nlargest(15, "Final_Prob")
            colors_bar = [SUBJECT_COLORS.get(s, "#94a3b8") for s in top15["Subject"]]
            fig2 = go.Figure(go.Bar(
                x=top15["Final_Prob"],
                y=top15["Subtopic"],
                orientation="h",
                marker_color=colors_bar,
                text=[f"{p:.0%}" for p in top15["Final_Prob"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
            fig2.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Top 15 highest-confidence topics", font=dict(size=13)),
                height=320,
                xaxis=dict(range=[0, 1.1], tickformat=".0%",
                           gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
                # FIXED: Combined yaxis settings into one dict to avoid keyword repetition
                yaxis=dict(autorange="reversed", gridcolor="#1e2a3a"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            subj_counts = forecast_df["Subject"].value_counts().reset_index()
            subj_counts.columns = ["Subject", "Count"]
            fig3 = go.Figure(go.Pie(
                labels=subj_counts["Subject"],
                values=subj_counts["Count"],
                hole=0.62,
                marker_colors=[SUBJECT_COLORS.get(s, "#94a3b8") for s in subj_counts["Subject"]],
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig3.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Forecast share by subject", font=dict(size=13)),
                height=280, showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with r2c2:
            bins   = [0, 0.5, 0.65, 0.75, 0.85, 0.95, 1.01]
            labels = ["<50%", "50–65%", "65–75%", "75–85%", "85–95%", "95%+"]
            forecast_df["Bucket"] = pd.cut(
                forecast_df["Final_Prob"], bins=bins, labels=labels, right=False
            )
            bucket_counts = forecast_df["Bucket"].value_counts().reindex(labels).fillna(0)
            fig4 = go.Figure(go.Bar(
                x=labels,
                y=bucket_counts.values,
                marker_color=["#1e2a3a","#1e3a4a","#1a4060","#1a5070","#1a6080","#38bdf8"],
                text=bucket_counts.values.astype(int),
                textposition="outside",
                textfont=dict(size=11, color="#94a3b8"),
            ))
            fig4.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Topics by confidence bucket", font=dict(size=13)),
                height=280,
                xaxis=dict(gridcolor="#1e2a3a"),
                yaxis=dict(gridcolor="#1e2a3a"),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3 — HISTORICAL TRENDS
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-header">Historical Pattern Analysis</div>', unsafe_allow_html=True)

    if historical_df.empty:
        st.info("Export `topic_yearly` from training script to unlock historical charts.")
    else:
        h1, h2 = st.columns(2)

        with h1:
            if "Year" in historical_df.columns and "Subject" in historical_df.columns:
                yearly = (historical_df.groupby(["Year", "Subject"])
                          .size().reset_index(name="Count"))
                fig_h1 = px.line(
                    yearly, x="Year", y="Count", color="Subject",
                    color_discrete_map=SUBJECT_COLORS,
                    markers=True, template="plotly_dark",
                    title="Yearly question volume by subject",
                )
                fig_h1.update_layout(**PLOT_LAYOUT, height=300, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
                st.plotly_chart(fig_h1, use_container_width=True)

        with h2:
            if "Is_Numerical" in historical_df.columns and "Subject" in historical_df.columns:
                num_dist = (historical_df.groupby(["Subject", "Is_Numerical"])
                            .size().unstack(fill_value=0).reset_index())
                num_dist.columns = ["Subject", "MCQ", "Numerical"]
                fig_h2 = go.Figure()
                for qtype, color in [("MCQ","#38bdf8"), ("Numerical","#f472b6")]:
                    if qtype in num_dist.columns:
                        fig_h2.add_trace(go.Bar(
                            name=qtype, x=num_dist["Subject"], y=num_dist[qtype],
                            marker_color=color,
                        ))
                fig_h2.update_layout(
                    **PLOT_LAYOUT, barmode="group", height=300, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                    title=dict(text="MCQ vs Numerical distribution", font=dict(size=13)),
                )
                st.plotly_chart(fig_h2, use_container_width=True)

        if "Chapter" in historical_df.columns:
            top_chaps = (historical_df["Chapter"].value_counts()
                         .head(15).reset_index())
            top_chaps.columns = ["Chapter", "Count"]
            fig_h3 = go.Figure(go.Bar(
                x=top_chaps["Count"], y=top_chaps["Chapter"],
                orientation="h",
                marker_color="#38bdf8",
                text=top_chaps["Count"],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
            # FIXED: Combined AXIS_STYLE with custom yaxis properties to avoid repetition
            fig_h3.update_layout(
                **PLOT_LAYOUT, height=380, xaxis=AXIS_STYLE,
                title=dict(text="Top 15 highest-frequency chapters", font=dict(size=13)),
                yaxis=dict(**AXIS_STYLE, autorange="reversed"),
            )
            st.plotly_chart(fig_h3, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4 — COMMUNITY
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="section-header">Community Strategy Feed</div>', unsafe_allow_html=True)

    with st.expander("Share your preparation notes"):
        with st.form("review_form", clear_on_submit=True):
            f_name    = st.text_input("Name / alias")
            f_rating  = st.select_slider(
                "How confident are you in these predictions?",
                options=[1, 2, 3, 4, 5], value=4,
            )
            f_comment = st.text_area("Your strategy or notes", height=100)
            if st.form_submit_button("Post", type="primary"):
                if f_name and f_comment:
                    add_review(f_name, f_rating, f_comment)
                    st.rerun()
                else:
                    st.warning("Name and comment are required.")

    reviews = get_reviews()
    if reviews.empty:
        st.info("No posts yet. Be the first to share your strategy.")
    else:
        for _, r in reviews.iterrows():
            stars = "★" * int(r["rating"]) + "☆" * (5 - int(r["rating"]))
            st.markdown(f"""
            <div class="review-card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
                <span style="font-weight:600;color:#38bdf8">{r['name']}</span>
                <span style="color:#fbbf24;font-size:0.85rem;letter-spacing:0.05em">{stars}</span>
              </div>
              <div style="color:#94a3b8;line-height:1.6;font-size:0.9rem">{r['comment']}</div>
              <div style="margin-top:0.7rem;font-size:0.72rem;color:#334155;font-family:'Space Mono',monospace">{r['date']}</div>
            </div>
            """, unsafe_allow_html=True)