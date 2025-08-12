"""Comprehensive churn analysis script.

This module provides a command line interface for performing an end-to-end
exploratory analysis on customer churn datasets. It handles missing value
inspection, feature distribution visualisation, correlation analysis,
clustering, dimensionality reduction, feature importance estimation and
survival analysis. The goal is to surface actionable insights and deliver
all artefacts to an output directory.

Example
-------
python churn_analysis.py path/to/churn.csv --output output
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional imports -----------------------------------------------------------
try:  # shap is optional and can be heavy
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully
    SHAP_AVAILABLE = False

try:  # lifelines for survival analysis
    from lifelines import KaplanMeierFitter

    LIFELINES_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully
    LIFELINES_AVAILABLE = False

# Silence common FutureWarnings from third-party libraries to keep console output clean
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def ensure_output_dir(path: str) -> None:
    """Create output directory if it does not exist."""

    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""

    df = pd.read_csv(csv_path)
    print(f"Loaded data with shape {df.shape}")
    print(df.head())
    return df


def report_missing_values(df: pd.DataFrame, output_dir: str) -> pd.Series:
    """Report missing values and save to HTML.

    Returns
    -------
    pd.Series
        Series with counts of missing values per column.
    """

    missing = df.isnull().sum()
    print("Missing values:\n", missing)
    strategies = {}
    for col, count in missing.items():
        if count == 0:
            continue
        if df[col].dtype.kind in "bifc":
            strategies[col] = "Consider imputing with mean/median."  # numeric
        else:
            strategies[col] = "Consider imputing with mode or dropping the column."
    miss_df = pd.DataFrame({"missing": missing, "suggestion": pd.Series(strategies)})
    miss_df.to_html(os.path.join(output_dir, "missing_values.html"))
    return missing


def summary_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """Compute summary statistics for numeric and categorical columns."""

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    num_summary = num_df.describe().T
    cat_summary = cat_df.describe(include="all").T

    num_summary.to_html(os.path.join(output_dir, "numeric_summary.html"))
    cat_summary.to_html(os.path.join(output_dir, "categorical_summary.html"))


def plot_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """Create histogram, boxplot and violin plots for each numeric column."""

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df[col].dropna(), ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        sns.violinplot(x=df[col], ax=axes[2])
        axes[2].set_title(f"Violin plot of {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close(fig)


def correlation_analysis(df: pd.DataFrame, output_dir: str, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Compute correlation matrix and flag highly collinear pairs."""

    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close(fig)

    high_corr = []
    for i, j in combinations(corr.columns, 2):
        if abs(corr.loc[i, j]) >= threshold:
            high_corr.append((i, j, corr.loc[i, j]))
    if high_corr:
        print("Highly collinear pairs:")
        for pair in high_corr:
            print(pair)
    pd.DataFrame(high_corr, columns=["feature_1", "feature_2", "corr"]).to_html(
        os.path.join(output_dir, "high_correlation_pairs.html")
    )
    return high_corr


def transform_skewed_features(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Detect skewed numeric features and apply log/Box–Cox transforms."""

    transformed = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        skewness = df[col].dropna().skew()
        if abs(skewness) > 1:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(df[col].dropna(), ax=ax[0])
            ax[0].set_title(f"{col} before")
            data = df[col].dropna()
            if (data <= 0).any():
                transformed[col] = np.log1p(data)
            else:
                transformed[col] = stats.boxcox(data)[0]
            sns.histplot(transformed[col].dropna(), ax=ax[1])
            ax[1].set_title(f"{col} after")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{col}_skew_transform.png"))
            plt.close(fig)
    return transformed


def clustering_analysis(df: pd.DataFrame, output_dir: str, target: str | None = "churn") -> None:
    """Run K-Means clustering and profile clusters against churn rate."""

    numeric = df.select_dtypes(include=np.number).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters
    ensure_output_dir(output_dir)
    df[["cluster"]].to_csv(os.path.join(output_dir, "cluster_assignments.csv"))
    if target and target in df.columns:
        profile = df.groupby("cluster")[target].mean()
        profile.to_frame("churn_rate").to_html(os.path.join(output_dir, "cluster_profile.html"))


def pca_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Perform PCA and plot explained variance."""

    numeric = df.select_dtypes(include=np.number).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric)
    pca = PCA()
    pca.fit(X_scaled)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained variance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_explained_variance.png"))
    plt.close(fig)


def compute_woe_iv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Calculate WOE and IV for each feature relative to target."""

    iv_list = []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if series.dtype.kind in "bifc":
            # bin numeric features
            series = pd.qcut(series.rank(method="first"), q=10, duplicates="drop")
        grouped = pd.DataFrame({col: series, target: df[target]}).groupby(col)
        total = grouped[target].count()
        event = grouped[target].sum()
        non_event = total - event
        dist_event = (event / event.sum()).replace(0, 1e-6)
        dist_non_event = (non_event / non_event.sum()).replace(0, 1e-6)
        woe = np.log(dist_event / dist_non_event)
        iv = ((dist_event - dist_non_event) * woe).sum()
        iv_list.append({"feature": col, "iv": iv})
    iv_df = pd.DataFrame(iv_list).sort_values("iv", ascending=False)
    return iv_df


def baseline_random_forest(df: pd.DataFrame, target: str, output_dir: str) -> Tuple[Pipeline, pd.DataFrame, List[str]]:
    """Train RandomForest, compute feature importances and SHAP/permutation."""

    y = df[target]
    X = df.drop(columns=[target])
    numeric = X.select_dtypes(include=np.number).columns
    categorical = X.select_dtypes(exclude=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(random_state=42))]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"RandomForest accuracy: {acc:.3f}")

    model = clf.named_steps["model"]
    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances.to_csv(os.path.join(output_dir, "feature_importances.csv"))

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(clf.named_steps["preprocessor"].transform(X_test))
        shap.summary_plot(shap_values[1], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary.png"))
        plt.close()
    else:  # fallback to permutation importance
        from sklearn.inspection import permutation_importance

        r = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
        perm_df = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
        perm_df.to_csv(os.path.join(output_dir, "permutation_importance.csv"))
    return clf, X_test, list(numeric)


def feature_interactions(clf: Pipeline, X: pd.DataFrame, features: Iterable[str], output_dir: str) -> None:
    """Analyse pairwise feature interactions via partial dependence plots."""

    for pair in combinations(list(features)[:3], 2):  # top 3 features
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(clf, X, [pair], ax=ax)
        fig.savefig(os.path.join(output_dir, f"pdp_{pair[0]}_{pair[1]}.png"))
        plt.close(fig)


def survival_analysis(df: pd.DataFrame, output_dir: str, tenure_col: str = "tenure", target: str = "churn") -> None:
    """Generate Kaplan–Meier survival curves if tenure column exists."""

    if not LIFELINES_AVAILABLE:
        print("lifelines not installed; skipping survival analysis")
        return
    if tenure_col not in df.columns or target not in df.columns:
        print("Required columns for survival analysis not found; skipping")
        return
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[tenure_col], event_observed=df[target])
    ax = kmf.plot()
    ax.set_title("Survival curve")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "survival_curve.png"))
    plt.close(fig)


def generate_markdown_summary(df: pd.DataFrame, output_dir: str, iv_df: pd.DataFrame | None = None) -> None:
    """Create a simple Markdown report summarising the analysis."""

    lines = ["# Churn Analysis Summary", ""]
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    if iv_df is not None:
        top_iv = iv_df.head(5)
        lines.append("## Top features by Information Value")
        lines.append(top_iv.to_markdown(index=False))
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive churn analysis")
    parser.add_argument("csv", help="Path to CSV data file")
    parser.add_argument("--output", default="output", help="Directory to store artefacts")
    args = parser.parse_args()

    ensure_output_dir(args.output)
    df = load_data(args.csv)
    report_missing_values(df, args.output)
    summary_statistics(df, args.output)
    plot_distributions(df, args.output)
    correlation_analysis(df, args.output)
    transformed_df = transform_skewed_features(df, args.output)
    clustering_analysis(transformed_df, args.output)
    pca_analysis(transformed_df, args.output)

    iv_df = None
    if "churn" in df.columns:
        iv_df = compute_woe_iv(df, "churn")
        iv_df.to_csv(os.path.join(args.output, "iv_scores.csv"), index=False)
        clf, X_test, numeric_features = baseline_random_forest(df, "churn", args.output)
        feature_interactions(clf, X_test, numeric_features, args.output)
        survival_analysis(df, args.output)
    generate_markdown_summary(df, args.output, iv_df)


if __name__ == "__main__":
    main()
