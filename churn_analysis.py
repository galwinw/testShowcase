"""Streamlit dashboard for comprehensive churn analysis.

This module exposes a `main` function that can be executed with
``streamlit run churn_analysis.py -- --csv_path <file>`` to launch an
interactive dashboard.  The original version of this script wrote every
plot and table to an ``output/`` directory; per the updated requirements all
analysis now renders directly inside the dashboard so the user can explore
the results without digging through files on disk.

The code remains modular with small helper functions for each analytical
step.  Each helper returns data or matplotlib figures that are immediately
displayed using Streamlit primitives such as ``st.write`` or ``st.pyplot``.

Only commonly available open–source libraries are used (``pandas``,
``numpy``, ``matplotlib``, ``seaborn``, ``scikit‑learn``, ``shap``, and
``lifelines``).  The script is designed to be run out of the box and does not
create any artefacts on the filesystem.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
from lifelines import KaplanMeierFitter


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and show a preview in the dashboard."""

    df = pd.read_csv(path)
    st.write(f"Loaded data with shape: {df.shape}")
    st.dataframe(df.head())
    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value report with simple strategy suggestions."""

    missing = df.isnull().sum()
    percent = missing / len(df) * 100
    report = pd.DataFrame({"missing_count": missing, "percent": percent})
    report["suggestion"] = np.where(
        report["percent"] < 5, "Impute", "Consider deletion/advanced imputation"
    )
    st.subheader("Missing Values")
    st.dataframe(report)
    return report


def summary_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Show summary statistics for numeric and categorical features."""

    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)

    num_stats = numeric.describe().T
    cat_stats = categorical.describe().T

    st.subheader("Summary Statistics – Numeric")
    st.dataframe(num_stats)
    st.subheader("Summary Statistics – Categorical")
    st.dataframe(cat_stats)
    return num_stats, cat_stats


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot distributions, boxplots, and violin plots for numeric columns."""

    st.subheader("Distributions")
    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df[col].dropna(), ax=axes[0], kde=True)
        axes[0].set_title(f"{col} distribution")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} boxplot")
        sns.violinplot(x=df[col], ax=axes[2])
        axes[2].set_title(f"{col} violin")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def correlation_analysis(df: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Compute correlation matrix, display heatmap, and return highly correlated pairs."""

    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()
    st.subheader("Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

    high_corr_pairs = [
        (i, j, corr.loc[i, j])
        for i in corr.columns
        for j in corr.columns
        if i < j and abs(corr.loc[i, j]) > threshold
    ]
    if high_corr_pairs:
        st.write("High correlations (>|{:.0%}|):".format(threshold))
        hc_df = pd.DataFrame(high_corr_pairs, columns=["feature_a", "feature_b", "corr"])
        st.dataframe(hc_df)
    else:
        st.write("No correlations above threshold detected.")
    return high_corr_pairs


def handle_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """Detect skewed features and apply log or Box–Cox transform with plots."""

    st.subheader("Skewness Correction")
    numeric = df.select_dtypes(include=np.number)
    skewness = numeric.apply(lambda x: x.dropna().skew())
    skewed_cols = skewness[skewness.abs() > 1].index
    pt = PowerTransformer(method="box-cox", standardize=False)
    for col in skewed_cols:
        series = df[col].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(series, ax=axes[0], kde=True)
        axes[0].set_title(f"{col} before")
        if (series <= 0).any():
            transformed = np.log1p(series - series.min() + 1)
            method = "log"
        else:
            transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
            method = "boxcox"
        sns.histplot(transformed, ax=axes[1], kde=True)
        axes[1].set_title(f"{col} after ({method})")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        df[f"{col}_{method}"] = pd.Series(transformed, index=series.index)
    if not skewed_cols.any():
        st.write("No significantly skewed features detected.")
    return df


def clustering_analysis(df: pd.DataFrame, target_col: str, n_clusters: int = 3) -> pd.DataFrame:
    """Run KMeans clustering and profile clusters against churn."""

    numeric = df.select_dtypes(include=np.number).drop(columns=[target_col], errors="ignore")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.fillna(0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    df["cluster"] = clusters

    if target_col in df.columns:
        profile = df.groupby("cluster")[target_col].mean().rename("churn_rate")
        st.subheader("Cluster Churn Rate")
        st.dataframe(profile)
        fig, ax = plt.subplots()
        sns.barplot(x=profile.index, y=profile.values, ax=ax)
        ax.set_ylabel("Churn Rate")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    return df


def pca_analysis(df: pd.DataFrame) -> None:
    """Run PCA on numeric features and plot explained variance."""

    numeric = df.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.fillna(0))
    pca = PCA()
    pca.fit(scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(explained) + 1), explained, marker="o")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    fig.tight_layout()
    st.subheader("PCA Explained Variance")
    st.pyplot(fig)
    plt.close(fig)


def woe_iv_analysis(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute weight-of-evidence (WOE) and information value (IV)."""

    st.subheader("WOE / IV Analysis")
    target = df[target_col]
    features = [c for c in df.columns if c != target_col]
    iv_records = []
    woe_records = []
    for col in features:
        series = df[col]
        if series.dtype.kind in "bifc":
            binned = pd.qcut(series, q=10, duplicates="drop")
        else:
            binned = series.astype(str)
        data = pd.DataFrame({"feature": binned, "target": target})
        grouped = data.groupby("feature")
        event = grouped["target"].sum() + 0.5
        non_event = grouped["target"].count() - grouped["target"].sum() + 0.5
        dist_event = event / event.sum()
        dist_non_event = non_event / non_event.sum()
        woe = np.log(dist_event / dist_non_event)
        iv = ((dist_event - dist_non_event) * woe).sum()
        iv_records.append({"feature": col, "iv": iv})
        for bin_name, w in woe.items():
            woe_records.append({"feature": col, "bin": str(bin_name), "woe": w})
    iv_df = pd.DataFrame(iv_records).sort_values(by="iv", ascending=False)
    st.dataframe(iv_df)
    return iv_df


def random_forest_analysis(df: pd.DataFrame, target_col: str) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """Train RandomForest, plot feature importances and SHAP values."""

    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.subheader("Random Forest Feature Importances")
    fig, ax = plt.subplots(figsize=(8, max(4, len(importances.head(20)) * 0.4)))
    sns.barplot(x=importances.head(20), y=importances.head(20).index, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, show=False)
    st.pyplot(bbox_inches="tight")

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean}).sort_values(
        by="importance", ascending=False
    )
    st.subheader("Permutation Importances")
    st.dataframe(perm_df)

    return model, X_test


def partial_dependence_analysis(model: RandomForestClassifier, X: pd.DataFrame) -> None:
    """Plot partial dependence for top two features and their interaction."""

    features = list(X.columns[:2])
    if len(features) >= 2:
        display = PartialDependenceDisplay.from_estimator(
            model, X, [features[0], features[1], (features[0], features[1])]
        )
        display.figure_.tight_layout()
        st.subheader("Partial Dependence")
        st.pyplot(display.figure_)
        plt.close(display.figure_)


def survival_analysis(df: pd.DataFrame, target_col: str, tenure_col: str) -> None:
    """Generate Kaplan–Meier survival curve if tenure column exists."""

    if tenure_col in df.columns:
        kmf = KaplanMeierFitter()
        kmf.fit(df[tenure_col], event_observed=df[target_col])
        ax = kmf.plot_survival_function()
        plt.tight_layout()
        st.subheader("Survival Curve")
        st.pyplot(ax.figure)
        plt.close(ax.figure)


def generate_summary(
    missing_report: pd.DataFrame,
    high_corr: List[Tuple[str, str, float]],
    iv_df: pd.DataFrame,
) -> None:
    """Display a markdown summary of key insights."""

    lines = ["# Churn Analysis Summary", ""]
    lines.append("## Missing Values")
    lines.append(missing_report.to_markdown())
    lines.append("\n## High Correlations")
    if high_corr:
        corr_lines = [f"- {a} & {b}: {v:.2f}" for a, b, v in high_corr]
        lines.extend(corr_lines)
    else:
        lines.append("No correlations above threshold detected.")
    lines.append("\n## Information Value")
    lines.append(iv_df.to_markdown(index=False))
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------------
# Streamlit entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive churn analysis tool")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--target-column", default="churn", help="Name of churn target column")
    parser.add_argument("--tenure-column", default="tenure", help="Name of tenure column if available")
    args, _ = parser.parse_known_args()

    st.title("Churn Analysis Dashboard")

    df = load_data(args.csv_path)
    target_col = args.target_column
    if target_col not in df.columns:
        alt = [c for c in df.columns if c.lower() == target_col.lower()]
        if alt:
            target_col = alt[0]
        else:
            st.error(f"Target column '{args.target_column}' not found in data")
            return

    missing_report = report_missing_values(df)
    summary_statistics(df)
    plot_distributions(df)
    high_corr = correlation_analysis(df)
    df = handle_skewness(df)
    df = clustering_analysis(df, target_col)
    pca_analysis(df)
    iv_df = woe_iv_analysis(df, target_col)
    rf_model, X_test = random_forest_analysis(df.dropna(subset=[target_col]), target_col)
    partial_dependence_analysis(rf_model, X_test)
    survival_analysis(df, target_col, args.tenure_column)
    generate_summary(missing_report, high_corr, iv_df)


if __name__ == "__main__":
    main()

