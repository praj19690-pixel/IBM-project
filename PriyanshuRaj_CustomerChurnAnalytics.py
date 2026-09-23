"""
AI-Powered Customer Churn Prediction & Retention Analytics
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
Student : Priyanshu Raj

Run:
    streamlit run PriyanshuRaj_CustomerChurnAnalytics.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
RAW_PATH  = BASE_DIR / "data" / "raw"  / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROC_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Load raw CSV, apply cleaning steps, return (clean_df, quality_report)."""
    raw = pd.read_csv(RAW_PATH)
    df  = raw.copy()

    quality = {
        "rows_raw"          : len(df),
        "cols_raw"          : df.shape[1],
        "nulls_raw"         : int(df.isnull().sum().sum()),
        "blanks_totalcharges": int((df["TotalCharges"].str.strip() == "").sum()),
        "duplicates_raw"    : int(df.duplicated().sum()),
    }

    # 1. Strip whitespace
    str_cols = [c for c in df.columns if df[c].dtype == object]
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2. TotalCharges: blank → 0.0, cast to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 3. SeniorCitizen: 0/1 → No/Yes
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 4. Drop customerID (identifier, no predictive value)
    df = df.drop(columns=["customerID"])

    quality.update({
        "rows_clean"   : len(df),
        "cols_clean"   : df.shape[1],
        "nulls_clean"  : int(df.isnull().sum().sum()),
        "duplicates_clean_no_id": int(df.duplicated().sum()),
        "cleaning_steps": [
            {
                "step"  : "Strip whitespace",
                "detail": "Applied str.strip() to all string columns",
                "impact": "Prevents silent category mismatches"
            },
            {
                "step"  : "Fix TotalCharges",
                "detail": "Cast str->float; 11 blanks (tenure=0) set to 0.0",
                "impact": "Enables numeric operations and ML"
            },
            {
                "step"  : "Normalise SeniorCitizen",
                "detail": "Mapped int 0/1 -> 'No'/'Yes'",
                "impact": "Consistent encoding with other binary columns"
            },
            {
                "step"  : "Drop customerID",
                "detail": "Unique identifier has no predictive signal",
                "impact": "Prevents data leakage and overfitting"
            },
        ]
    })

    # Save processed file if not already saved
    PROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC_PATH, index=False)

    return df, quality


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Tenure group
    bins   = [-1, 12, 24, 48, 72]
    labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # 2. Monthly charge band
    charge_bins   = [0, 35, 65, 90, 120]
    charge_labels = ["Low (<35)", "Medium (35-65)", "High (65-90)", "Very High (>90)"]
    df["ChargeBand"] = pd.cut(df["MonthlyCharges"], bins=charge_bins, labels=charge_labels)

    # 3. Service count (internet-dependent services: security, backup, protection, support, streaming x2)
    internet_services = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                         "TechSupport","StreamingTV","StreamingMovies"]
    df["ServiceCount"] = sum(
        (df[c] == "Yes").astype(int) for c in internet_services
    )

    # 4. Has phone and internet
    df["HasPhone"]    = (df["PhoneService"] == "Yes").astype(int)
    df["HasInternet"] = (df["InternetService"] != "No").astype(int)

    # 5. Charge per tenure month (avoid div/0 for tenure=0)
    df["ChargePerTenure"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – KPI CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_kpis(df: pd.DataFrame) -> dict:
    total     = len(df)
    churned   = int((df["Churn"] == "Yes").sum())
    retained  = total - churned
    churn_pct = churned / total * 100

    return {
        "total_customers"   : total,
        "churned"           : churned,
        "retained"          : retained,
        "churn_rate"        : round(churn_pct, 2),
        "retention_rate"    : round(100 - churn_pct, 2),
        "avg_tenure"        : round(df["tenure"].mean(), 2),
        "avg_monthly"       : round(df["MonthlyCharges"].mean(), 2),
        "avg_total"         : round(df["TotalCharges"].mean(), 2),
        "avg_monthly_churned"   : round(df.loc[df["Churn"]=="Yes","MonthlyCharges"].mean(), 2),
        "avg_monthly_retained"  : round(df.loc[df["Churn"]=="No","MonthlyCharges"].mean(), 2),
        "avg_tenure_churned"    : round(df.loc[df["Churn"]=="Yes","tenure"].mean(), 2),
        "avg_tenure_retained"   : round(df.loc[df["Churn"]=="No","tenure"].mean(), 2),
        "avg_total_churned"     : round(df.loc[df["Churn"]=="Yes","TotalCharges"].mean(), 2),
        "avg_total_retained"    : round(df.loc[df["Churn"]=="No","TotalCharges"].mean(), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train three classifiers. Returns (results_dict, best_model_name, preprocessor)."""

    df_ml = df.copy()
    df_ml["Churn_bin"] = (df_ml["Churn"] == "Yes").astype(int)

    # Feature set – drop target and engineered categorical columns that are
    # redundant with their source numeric columns in tree models
    drop_cols = ["Churn", "Churn_bin", "TenureGroup", "ChargeBand"]
    feature_cols = [c for c in df_ml.columns if c not in drop_cols]

    X = df_ml[feature_cols]
    y = df_ml["Churn_bin"]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest"      : RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results   = {}
    pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_prob  = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            "accuracy" : round(accuracy_score(y_test, y_pred)      * 100, 2),
            "precision": round(precision_score(y_test, y_pred)      * 100, 2),
            "recall"   : round(recall_score(y_test, y_pred)         * 100, 2),
            "f1"       : round(f1_score(y_test, y_pred)             * 100, 2),
            "roc_auc"  : round(roc_auc_score(y_test, y_prob)        * 100, 2),
            "cm"       : cm,
            "fpr"      : fpr,
            "tpr"      : tpr,
            "report"   : classification_report(y_test, y_pred, target_names=["No Churn","Churn"]),
        }
        pipelines[name] = pipe

    # Best model = highest ROC-AUC
    best_name = max(results, key=lambda n: results[n]["roc_auc"])

    return results, best_name, pipelines, feature_cols, num_cols, cat_cols, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(pipelines, feature_cols, num_cols, cat_cols):
    importance_dict = {}

    for name, pipe in pipelines.items():
        prep = pipe.named_steps["prep"]
        clf  = pipe.named_steps["clf"]

        # Reconstruct feature names after OHE
        ohe      = prep.named_transformers_["cat"]
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        all_feat  = num_cols + ohe_names

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            continue

        imp_df = pd.DataFrame({"Feature": all_feat, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(20)
        importance_dict[name] = imp_df

    return importance_dict


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – AI INSIGHTS (rule-based, derived from actual computed values)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ai_insights(df: pd.DataFrame, kpis: dict, ml_results: dict, best_model: str) -> dict:
    """Generate analytical insights derived entirely from actual computed values.
    No external API is used. All statements are traceable to calculated data."""

    insights = {}

    # ── Churn pattern insights ────────────────────────────────────────────────
    churn_patterns = []

    # Contract
    ct = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(2)
    highest_contract = ct.idxmax()
    lowest_contract  = ct.idxmin()
    churn_patterns.append(
        f"Customers on '{highest_contract}' contracts show a churn rate of {ct[highest_contract]:.1f}%, "
        f"compared with {ct[lowest_contract]:.1f}% for '{lowest_contract}' contracts. "
        f"This pattern suggests contract length is associated with customer retention."
    )

    # Tenure groups
    tg = df.groupby("TenureGroup", observed=False)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).round(2)
    highest_tg = tg.idxmax()
    churn_patterns.append(
        f"The tenure group '{highest_tg}' shows the highest observed churn rate of {tg[highest_tg]:.1f}%. "
        f"Customers with longer tenure tend to show lower observed churn rates, "
        f"which may reflect stronger loyalty or higher switching costs over time."
    )

    # Internet service
    isp = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(2)
    if "Fiber optic" in isp.index:
        churn_patterns.append(
            f"Fiber optic internet customers show an observed churn rate of {isp['Fiber optic']:.1f}%, "
            f"higher than DSL ({isp.get('DSL', 0):.1f}%) and customers with no internet service "
            f"({isp.get('No', 0):.1f}%). This association may warrant further investigation."
        )

    # Payment method
    pm = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(2)
    highest_pm = pm.idxmax()
    lowest_pm  = pm.idxmin()
    churn_patterns.append(
        f"Among payment methods, '{highest_pm}' is associated with the highest observed churn rate "
        f"({pm[highest_pm]:.1f}%), while '{lowest_pm}' shows the lowest ({pm[lowest_pm]:.1f}%)."
    )

    insights["Churn Pattern Insights"] = churn_patterns

    # ── Customer characteristics ──────────────────────────────────────────────
    cust_insights = []

    avg_tenure_diff = kpis["avg_tenure_retained"] - kpis["avg_tenure_churned"]
    cust_insights.append(
        f"Churned customers have an average tenure of {kpis['avg_tenure_churned']} months, "
        f"compared with {kpis['avg_tenure_retained']} months for retained customers "
        f"(a difference of {avg_tenure_diff:.1f} months)."
    )

    avg_charge_diff = kpis["avg_monthly_churned"] - kpis["avg_monthly_retained"]
    cust_insights.append(
        f"Churned customers have an average monthly charge of ${kpis['avg_monthly_churned']}, "
        f"compared with ${kpis['avg_monthly_retained']} for retained customers "
        f"(a difference of ${avg_charge_diff:.2f})."
    )

    senior = df.groupby("SeniorCitizen")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(2)
    if "Yes" in senior.index and "No" in senior.index:
        cust_insights.append(
            f"Senior citizens show an observed churn rate of {senior['Yes']:.1f}%, "
            f"compared with {senior['No']:.1f}% for non-senior customers."
        )

    insights["Customer Characteristics"] = cust_insights

    # ── Model insights ────────────────────────────────────────────────────────
    model_insights = []
    for model_name, res in ml_results.items():
        model_insights.append(
            f"{model_name}: Accuracy {res['accuracy']}%, Precision {res['precision']}%, "
            f"Recall {res['recall']}%, F1 {res['f1']}%, ROC-AUC {res['roc_auc']}%."
        )
    best_res = ml_results[best_model]
    model_insights.append(
        f"Among the evaluated models, {best_model} achieved the highest ROC-AUC of "
        f"{best_res['roc_auc']}%, making it the best discriminator of churn vs. non-churn "
        f"on the held-out test set."
    )
    insights["Model Performance Insights"] = model_insights

    # ── Business implications ─────────────────────────────────────────────────
    biz = []
    biz.append(
        f"The overall churn rate is {kpis['churn_rate']}% ({kpis['churned']:,} out of "
        f"{kpis['total_customers']:,} customers). This represents a meaningful customer loss "
        f"that retention programs could target."
    )
    biz.append(
        f"Month-to-month contract customers (churn rate: {ct.get('Month-to-month', 0):.1f}%) "
        f"represent the highest-risk segment. Offering incentives to upgrade to annual or "
        f"two-year contracts could be associated with improved retention, based on observed patterns."
    )
    biz.append(
        "Customers with shorter tenure show higher observed churn rates. Early engagement "
        "and onboarding programs targeting customers in their first 12 months may be "
        "worth evaluating."
    )
    biz.append(
        "Note: All insights are based on observed associations in the dataset. "
        "Correlation does not imply causation. Business decisions should incorporate "
        "additional domain knowledge and controlled experiments."
    )
    insights["Business Implications"] = biz

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – PREDICTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def predict_churn(pipeline, input_dict: dict, feature_cols: list) -> tuple[str, float]:
    """Run a single prediction through the trained pipeline."""
    row = pd.DataFrame([input_dict], columns=feature_cols)
    pred  = pipeline.predict(row)[0]
    prob  = pipeline.predict_proba(row)[0][1]
    label = "Yes (Likely to Churn)" if pred == 1 else "No (Likely to Stay)"
    return label, round(prob * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def apply_filters(df: pd.DataFrame, sidebar_filters: dict) -> pd.DataFrame:
    fdf = df.copy()
    if sidebar_filters.get("contract"):
        fdf = fdf[fdf["Contract"].isin(sidebar_filters["contract"])]
    if sidebar_filters.get("internet"):
        fdf = fdf[fdf["InternetService"].isin(sidebar_filters["internet"])]
    if sidebar_filters.get("payment"):
        fdf = fdf[fdf["PaymentMethod"].isin(sidebar_filters["payment"])]
    if sidebar_filters.get("churn_status"):
        fdf = fdf[fdf["Churn"].isin(sidebar_filters["churn_status"])]
    t_min, t_max = sidebar_filters.get("tenure_range", (0, 72))
    fdf = fdf[(fdf["tenure"] >= t_min) & (fdf["tenure"] <= t_max)]
    c_min, c_max = sidebar_filters.get("charge_range", (0, 200))
    fdf = fdf[(fdf["MonthlyCharges"] >= c_min) & (fdf["MonthlyCharges"] <= c_max)]
    return fdf


def kpi_cards(kpis: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers",    f"{kpis['total_customers']:,}")
    c2.metric("Churned",            f"{kpis['churned']:,}",    delta=f"{kpis['churn_rate']}%",     delta_color="inverse")
    c3.metric("Retained",           f"{kpis['retained']:,}",   delta=f"{kpis['retention_rate']}%", delta_color="normal")
    c4.metric("Avg Tenure (mo)",    f"{kpis['avg_tenure']}")
    c5.metric("Avg Monthly ($)",    f"{kpis['avg_monthly']}")
    c6.metric("Avg Total ($)",      f"{kpis['avg_total']:,}")


# ── Page renderers ─────────────────────────────────────────────────────────────

def render_overview(df, kpis):
    st.header("Executive Overview")
    kpi_cards(kpis)
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Churn distribution donut
    with col1:
        fig = px.pie(
            names=["Retained", "Churned"],
            values=[kpis["retained"], kpis["churned"]],
            hole=0.5,
            color_discrete_sequence=["#2563eb", "#ef4444"],
            title="Overall Churn Distribution"
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Churn by contract
    with col2:
        ct = df.groupby("Contract")["Churn"].value_counts(normalize=True).mul(100).round(2).reset_index(name="Pct")
        fig2 = px.bar(ct, x="Contract", y="Pct", color="Churn",
                      barmode="group", title="Churn Rate by Contract Type (%)",
                      color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                      labels={"Pct": "Percentage (%)"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Churn by tenure group
    with col3:
        tg = df.groupby("TenureGroup", observed=False)["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).round(2).reset_index(name="ChurnRate")
        fig3 = px.bar(tg, x="TenureGroup", y="ChurnRate",
                      title="Churn Rate by Tenure Group (%)",
                      color="ChurnRate", color_continuous_scale="RdYlGn_r",
                      labels={"ChurnRate": "Churn Rate (%)"})
        st.plotly_chart(fig3, use_container_width=True)

    # Monthly charges distribution
    with col4:
        fig4 = px.box(df, x="Churn", y="MonthlyCharges",
                      color="Churn",
                      color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                      title="Monthly Charges Distribution by Churn Status",
                      labels={"MonthlyCharges": "Monthly Charges ($)"})
        st.plotly_chart(fig4, use_container_width=True)

    # Key insights
    st.subheader("Key Observations")
    st.info(
        f"- Overall churn rate: **{kpis['churn_rate']}%** "
        f"({kpis['churned']:,} churned out of {kpis['total_customers']:,} customers)\n"
        f"- Churned customers have a lower average tenure "
        f"(**{kpis['avg_tenure_churned']} mo** vs **{kpis['avg_tenure_retained']} mo** for retained)\n"
        f"- Churned customers have higher average monthly charges "
        f"(**${kpis['avg_monthly_churned']}** vs **${kpis['avg_monthly_retained']}** for retained)\n"
        f"- Class balance: **{kpis['retention_rate']}%** retained / **{kpis['churn_rate']}%** churned"
    )


def render_churn_analysis(df):
    st.header("Churn Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Churn by internet service
        isp = df.groupby("InternetService")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).round(2).reset_index(name="ChurnRate")
        fig = px.bar(isp, x="InternetService", y="ChurnRate",
                     color="ChurnRate", color_continuous_scale="Reds",
                     title="Churn Rate by Internet Service (%)",
                     labels={"ChurnRate": "Churn Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by payment method
        pm = df.groupby("PaymentMethod")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).round(2).reset_index(name="ChurnRate")
        fig2 = px.bar(pm, x="PaymentMethod", y="ChurnRate",
                      color="ChurnRate", color_continuous_scale="Reds",
                      title="Churn Rate by Payment Method (%)",
                      labels={"ChurnRate": "Churn Rate (%)"},
                      text_auto=".1f")
        fig2.update_xaxes(tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Churn by senior citizen
        sc = df.groupby("SeniorCitizen")["Churn"].value_counts().reset_index(name="Count")
        fig3 = px.bar(sc, x="SeniorCitizen", y="Count", color="Churn",
                      barmode="group",
                      color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                      title="Churn Count by Senior Citizen Status",
                      labels={"SeniorCitizen": "Senior Citizen"})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Churn by paperless billing
        pb = df.groupby("PaperlessBilling")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).round(2).reset_index(name="ChurnRate")
        fig4 = px.bar(pb, x="PaperlessBilling", y="ChurnRate",
                      color="ChurnRate", color_continuous_scale="Reds",
                      title="Churn Rate by Paperless Billing (%)",
                      labels={"ChurnRate": "Churn Rate (%)"})
        st.plotly_chart(fig4, use_container_width=True)

    # Service-level churn table
    st.subheader("Churn Rate by Individual Service")
    service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    rows = []
    for col in service_cols:
        for val in df[col].unique():
            sub = df[df[col] == val]
            rate = (sub["Churn"] == "Yes").mean() * 100
            rows.append({"Service": col, "Value": val, "Customers": len(sub),
                         "Churn Rate (%)": round(rate, 2)})
    svc_df = pd.DataFrame(rows).sort_values(["Service","Value"])
    st.dataframe(svc_df, use_container_width=True, hide_index=True)


def render_customer_service(df, kpis):
    st.header("Customer & Service Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="tenure", color="Churn",
                           nbins=72, barmode="overlay", opacity=0.7,
                           color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                           title="Tenure Distribution by Churn Status",
                           labels={"tenure": "Tenure (months)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                          color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                          opacity=0.5,
                          title="Tenure vs Monthly Charges (coloured by Churn)",
                          labels={"tenure": "Tenure (months)",
                                  "MonthlyCharges": "Monthly Charges ($)"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(df, x="TenureGroup", y="MonthlyCharges", color="Churn",
                      color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                      title="Monthly Charges by Tenure Group & Churn",
                      labels={"MonthlyCharges": "Monthly Charges ($)",
                               "TenureGroup": "Tenure Group"},
                      category_orders={"TenureGroup": ["0-12 mo","13-24 mo","25-48 mo","49-72 mo"]})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(df, x="Churn", y="TotalCharges", color="Churn",
                      color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                      title="Total Charges Distribution by Churn Status",
                      labels={"TotalCharges": "Total Charges ($)"})
        st.plotly_chart(fig4, use_container_width=True)

    # Service count
    st.subheader("Service Subscription Count vs Churn")
    svc_ct = df.groupby(["ServiceCount","Churn"]).size().reset_index(name="Count")
    fig5 = px.bar(svc_ct, x="ServiceCount", y="Count", color="Churn",
                  barmode="group",
                  color_discrete_map={"Yes": "#ef4444", "No": "#2563eb"},
                  title="Churn by Number of Subscribed Services",
                  labels={"ServiceCount": "Number of Subscribed Services (of 6)"})
    st.plotly_chart(fig5, use_container_width=True)

    # Partner / Dependents table
    st.subheader("Churn by Demographics")
    demo_rows = []
    for col in ["gender","Partner","Dependents","SeniorCitizen"]:
        for val in sorted(df[col].unique()):
            sub = df[df[col] == val]
            rate = (sub["Churn"] == "Yes").mean() * 100
            demo_rows.append({"Variable": col, "Value": val,
                               "Customers": len(sub), "Churn Rate (%)": round(rate, 2)})
    st.dataframe(pd.DataFrame(demo_rows), use_container_width=True, hide_index=True)


def render_machine_learning(ml_results, best_model, X_test, y_test, pipelines):
    st.header("Machine Learning")

    st.subheader("Methodology")
    st.markdown("""
- **Train/Test Split:** 80% train / 20% test (stratified on target)
- **Random State:** 42 (reproducible)
- **Preprocessing:** `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- **Pipeline:** Preprocessing + classifier in a single `sklearn` Pipeline
- **Models trained:** Logistic Regression, Random Forest, Gradient Boosting
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
    """)

    # Model comparison table
    st.subheader("Model Comparison")
    comp = []
    for name, res in ml_results.items():
        comp.append({
            "Model"     : name,
            "Accuracy %": res["accuracy"],
            "Precision %": res["precision"],
            "Recall %"  : res["recall"],
            "F1 %"      : res["f1"],
            "ROC-AUC %": res["roc_auc"],
        })
    comp_df = pd.DataFrame(comp).set_index("Model")
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#bbf7d0"), use_container_width=True)

    st.info(
        f"**Best model by ROC-AUC:** {best_model} "
        f"({ml_results[best_model]['roc_auc']}%)\n\n"
        "ROC-AUC is chosen as the primary metric because the classes are imbalanced "
        "and we care about ranking churn risk, not just raw accuracy."
    )

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(ml_results))
    for i, (name, res) in enumerate(ml_results.items()):
        cm = res["cm"]
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            x=["Pred: No Churn","Pred: Churn"],
            y=["True: No Churn","True: Churn"],
            title=name
        )
        cols[i].plotly_chart(fig, use_container_width=True)

    # ROC curves
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="grey"))
    colors = ["#2563eb", "#16a34a", "#d97706"]
    for i, (name, res) in enumerate(ml_results.items()):
        fig_roc.add_trace(go.Scatter(
            x=res["fpr"], y=res["tpr"],
            mode="lines", name=f"{name} (AUC={res['roc_auc']}%)",
            line=dict(color=colors[i], width=2)
        ))
    fig_roc.update_layout(
        title="ROC Curves – All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.5, y=0.1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    sel = st.selectbox("Select model for classification report", list(ml_results.keys()))
    st.code(ml_results[sel]["report"], language="text")


def render_prediction(pipelines, feature_cols, num_cols, cat_cols, best_model, df):
    st.header("Churn Prediction")
    st.markdown(
        "Enter customer characteristics below. The model will output a predicted churn class "
        "and model-estimated churn probability."
    )
    st.warning(
        "**Disclaimer:** Predictions are probabilistic estimates based on patterns learned "
        "from training data. They are NOT guaranteed outcomes. Always combine model outputs "
        "with domain knowledge before making business decisions."
    )

    # Build input form using actual model features
    input_data = {}

    with st.form("prediction_form"):
        st.subheader("Account & Demographics")
        c1, c2, c3 = st.columns(3)
        input_data["gender"]          = c1.selectbox("Gender",         sorted(df["gender"].unique()))
        input_data["SeniorCitizen"]   = c2.selectbox("Senior Citizen", sorted(df["SeniorCitizen"].unique()))
        input_data["Partner"]         = c3.selectbox("Partner",        sorted(df["Partner"].unique()))
        c4, c5 = st.columns(2)
        input_data["Dependents"]      = c4.selectbox("Dependents",     sorted(df["Dependents"].unique()))
        input_data["tenure"]          = c5.slider("Tenure (months)", 0, 72, 12)

        st.subheader("Services")
        c6, c7, c8 = st.columns(3)
        input_data["PhoneService"]    = c6.selectbox("Phone Service",   sorted(df["PhoneService"].unique()))
        input_data["MultipleLines"]   = c7.selectbox("Multiple Lines",  sorted(df["MultipleLines"].unique()))
        input_data["InternetService"] = c8.selectbox("Internet Service",sorted(df["InternetService"].unique()))

        c9, c10, c11 = st.columns(3)
        input_data["OnlineSecurity"]  = c9.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
        input_data["OnlineBackup"]    = c10.selectbox("Online Backup",  sorted(df["OnlineBackup"].unique()))
        input_data["DeviceProtection"]= c11.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))

        c12, c13, c14 = st.columns(3)
        input_data["TechSupport"]     = c12.selectbox("Tech Support",   sorted(df["TechSupport"].unique()))
        input_data["StreamingTV"]     = c13.selectbox("Streaming TV",   sorted(df["StreamingTV"].unique()))
        input_data["StreamingMovies"] = c14.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))

        st.subheader("Account & Billing")
        c15, c16, c17 = st.columns(3)
        input_data["Contract"]        = c15.selectbox("Contract",        sorted(df["Contract"].unique()))
        input_data["PaperlessBilling"]= c16.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
        input_data["PaymentMethod"]   = c17.selectbox("Payment Method",  sorted(df["PaymentMethod"].unique()))

        c18, c19 = st.columns(2)
        input_data["MonthlyCharges"]  = c18.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.05)
        input_data["TotalCharges"]    = c19.number_input("Total Charges ($)", 0.0, 10000.0,
                                                          float(input_data["MonthlyCharges"] * input_data["tenure"]),
                                                          step=1.0)

        # Engineered features expected by the pipeline
        input_data["ServiceCount"]    = sum([
            1 for svc in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                           "TechSupport","StreamingTV","StreamingMovies"]
            if input_data.get(svc) == "Yes"
        ])
        input_data["HasPhone"]        = 1 if input_data["PhoneService"] == "Yes" else 0
        input_data["HasInternet"]     = 1 if input_data["InternetService"] != "No" else 0
        input_data["ChargePerTenure"] = (
            input_data["TotalCharges"] / input_data["tenure"]
            if input_data["tenure"] > 0
            else input_data["MonthlyCharges"]
        )

        model_choice = st.selectbox("Model to use for prediction", list(pipelines.keys()),
                                    index=list(pipelines.keys()).index(best_model))
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Filter to only features the pipeline was trained on
        row_input = {k: input_data[k] for k in feature_cols if k in input_data}
        label, prob = predict_churn(pipelines[model_choice], row_input, feature_cols)

        col_r1, col_r2 = st.columns(2)
        if "Yes" in label:
            col_r1.error(f"**Predicted Class:** {label}")
        else:
            col_r1.success(f"**Predicted Class:** {label}")
        col_r2.metric("Model-predicted churn probability", f"{prob}%")

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Churn Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ef4444" if prob > 50 else "#2563eb"},
                "steps": [
                    {"range": [0,  30], "color": "#bbf7d0"},
                    {"range": [30, 60], "color": "#fef08a"},
                    {"range": [60,100], "color": "#fecaca"},
                ],
                "threshold": {"line": {"color": "black","width": 4}, "value": 50}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)


def render_feature_importance(importance_dict):
    st.header("Feature Importance")
    st.info(
        "Feature importance indicates which features contributed most to each model's "
        "predictions on the training data. **Higher importance = stronger contribution to "
        "the model's decision.** This reflects statistical association within the model — "
        "it does NOT establish causal relationships."
    )

    model_sel = st.selectbox("Select model", list(importance_dict.keys()))
    imp_df = importance_dict[model_sel].head(15)

    fig = px.bar(imp_df.sort_values("Importance"),
                 x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title=f"Top 15 Feature Importances — {model_sel}",
                 labels={"Importance": "Importance Score"})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Feature Importance Table")
    st.dataframe(imp_df, use_container_width=True, hide_index=True)


def render_ai_insights(insights):
    st.header("AI-Based Analytical Insights")
    st.info(
        "All insights below are generated programmatically from actual computed values "
        "(churn rates, KPIs, model metrics, feature importances). No external AI API is used. "
        "Statements use associative language — correlation does not imply causation."
    )

    for section, items in insights.items():
        st.subheader(section)
        for item in items:
            st.markdown(f"- {item}")
        st.markdown("")


def render_data_quality(df_raw, quality: dict):
    st.header("Data Quality Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (Raw)",          f"{quality['rows_raw']:,}")
    col2.metric("Rows (After Clean)",  f"{quality['rows_clean']:,}")
    col3.metric("Nulls (Raw)",         f"{quality['nulls_raw']}")
    col4.metric("Nulls (After Clean)", f"{quality['nulls_clean']}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Blank TotalCharges",  f"{quality['blanks_totalcharges']}")
    col6.metric("Blank (After Clean)", "0")
    col7.metric("Duplicates (Raw)",    f"{quality['duplicates_raw']}")
    col8.metric("Cols After Clean",    f"{quality['cols_clean']}")

    st.subheader("Cleaning Operations")
    for step in quality["cleaning_steps"]:
        with st.expander(f"Step: {step['step']}"):
            st.markdown(f"**Detail:** {step['detail']}")
            st.markdown(f"**Impact:** {step['impact']}")

    st.subheader("Target Distribution")
    total = quality["rows_clean"]
    churned = quality.get("churned", 0)
    fig = px.bar(
        x=["No (Retained)", "Yes (Churned)"],
        y=[total - churned, churned],
        color=["No (Retained)", "Yes (Churned)"],
        color_discrete_map={"No (Retained)": "#2563eb", "Yes (Churned)": "#ef4444"},
        title="Target Variable Distribution (Post-Cleaning)",
        labels={"x": "Churn Status", "y": "Customer Count"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Quality Observations")
    st.markdown("""
- No true null (NaN) values existed in the raw dataset.
- `TotalCharges` was loaded as a string due to 11 blank entries — corrected to float.
- The 11 blank `TotalCharges` rows all have `tenure = 0` (brand-new customers with no charges yet); set to 0.0.
- `SeniorCitizen` was encoded as integer 0/1 — normalised to Yes/No for pipeline consistency.
- `customerID` was dropped: it is a unique identifier with no predictive value.
- No exact duplicate rows existed in the raw file (22 feature-level duplicates are real distinct customers).
- All numeric columns are within plausible ranges. No extreme outliers requiring removal were detected.
    """)

    st.subheader("Raw Dataset Sample (first 10 rows)")
    st.dataframe(df_raw.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – MAIN APP ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Churn Analytics | Priyanshu Raj",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Header ─────────────────────────────────────────────────────────────────
    st.title("AI-Powered Customer Churn Prediction & Retention Analytics")
    st.markdown("**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**  |  Student: **Priyanshu Raj**")
    st.markdown("---")

    # ── Load & check raw file ───────────────────────────────────────────────────
    if not RAW_PATH.exists():
        st.error(
            f"Dataset not found at `{RAW_PATH}`.\n\n"
            "Please download the Telco Customer Churn dataset from "
            "[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
            "and place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`."
        )
        st.stop()

    # ── Data loading & cleaning ─────────────────────────────────────────────────
    with st.spinner("Loading and cleaning data…"):
        df_raw       = load_raw()
        df_clean, quality = load_and_clean()

    # ── Feature engineering ─────────────────────────────────────────────────────
    with st.spinner("Engineering features…"):
        df = engineer_features(df_clean)

    # ── KPIs ────────────────────────────────────────────────────────────────────
    kpis_full = calculate_kpis(df)
    quality["churned"] = kpis_full["churned"]

    # ── ML (cached) ─────────────────────────────────────────────────────────────
    with st.spinner("Training machine learning models (first run only, then cached)…"):
        ml_results, best_model, pipelines, feature_cols, num_cols, cat_cols, X_test, y_test = train_models(df)

    importance_dict = get_feature_importance(pipelines, feature_cols, num_cols, cat_cols)
    insights        = generate_ai_insights(df, kpis_full, ml_results, best_model)

    # ── Sidebar filters ─────────────────────────────────────────────────────────
    st.sidebar.header("Filters")
    st.sidebar.caption("Filters apply to Overview, Churn Analysis, and Customer & Service pages.")

    sel_contract = st.sidebar.multiselect(
        "Contract Type", df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist()
    )
    sel_internet = st.sidebar.multiselect(
        "Internet Service", df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist()
    )
    sel_payment = st.sidebar.multiselect(
        "Payment Method", df["PaymentMethod"].unique().tolist(),
        default=df["PaymentMethod"].unique().tolist()
    )
    sel_churn = st.sidebar.multiselect(
        "Churn Status", ["Yes", "No"], default=["Yes", "No"]
    )
    tenure_range = st.sidebar.slider("Tenure Range (months)", 0, 72, (0, 72))
    charge_range = st.sidebar.slider("Monthly Charges Range ($)", 0, 150, (0, 150))

    sidebar_filters = {
        "contract"    : sel_contract,
        "internet"    : sel_internet,
        "payment"     : sel_payment,
        "churn_status": sel_churn,
        "tenure_range": tenure_range,
        "charge_range": charge_range,
    }

    df_filtered = apply_filters(df, sidebar_filters)

    if df_filtered.empty:
        st.warning("No data matches the current filters. Please adjust your sidebar selections.")
        st.stop()

    kpis_filtered = calculate_kpis(df_filtered)

    # ── Navigation ──────────────────────────────────────────────────────────────
    pages = [
        "Executive Overview",
        "Churn Analysis",
        "Customer & Service Analysis",
        "Machine Learning",
        "Churn Prediction",
        "Feature Importance",
        "AI Insights",
        "Data Quality",
    ]
    page = st.sidebar.radio("Navigate", pages)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Dataset:** {kpis_filtered['total_customers']:,} customers shown  \n"
        f"**Churn Rate (filtered):** {kpis_filtered['churn_rate']}%"
    )

    # ── Page routing ────────────────────────────────────────────────────────────
    if page == "Executive Overview":
        render_overview(df_filtered, kpis_filtered)

    elif page == "Churn Analysis":
        render_churn_analysis(df_filtered)

    elif page == "Customer & Service Analysis":
        render_customer_service(df_filtered, kpis_filtered)

    elif page == "Machine Learning":
        render_machine_learning(ml_results, best_model, X_test, y_test, pipelines)

    elif page == "Churn Prediction":
        render_prediction(pipelines, feature_cols, num_cols, cat_cols, best_model, df_clean)

    elif page == "Feature Importance":
        render_feature_importance(importance_dict)

    elif page == "AI Insights":
        render_ai_insights(insights)

    elif page == "Data Quality":
        render_data_quality(df_raw, quality)


if __name__ == "__main__":
    main()
