import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
from lifelines import KaplanMeierFitter


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and print basic info."""
    df = pd.read_csv(path)
    print(f"Loaded data with shape: {df.shape}")
    print(df.head())
    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value report with strategy suggestions."""
    missing = df.isnull().sum()
    percent = missing / len(df) * 100
    report = pd.DataFrame({"missing_count": missing, "percent": percent})
    report["suggestion"] = np.where(
        report["percent"] < 5, "Impute", "Consider deletion/advanced imputation"
    )
    print(report)
    return report


def summary_statistics(df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save summary stats for numeric and categorical features."""
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)

    num_stats = numeric.describe().T
    cat_stats = categorical.describe().T

    num_stats.to_html(os.path.join(output_dir, "numeric_summary.html"))
    cat_stats.to_html(os.path.join(output_dir, "categorical_summary.html"))

    return num_stats, cat_stats


def plot_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """Plot distributions, boxplots, and violin plots for numeric columns."""
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
        fig.savefig(os.path.join(output_dir, f"{col}_dist.png"))
        plt.close(fig)


def correlation_analysis(
    df: pd.DataFrame, output_dir: str, threshold: float = 0.8
) -> List[Tuple[str, str, float]]:
    """Compute correlation matrix, save heatmap, and return highly correlated pairs."""
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    high_corr_pairs = [
        (i, j, corr.loc[i, j])
        for i in corr.columns
        for j in corr.columns
        if i < j and abs(corr.loc[i, j]) > threshold
    ]
    return high_corr_pairs


def handle_skewness(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Detect skewed features and apply log or Box-Cox transform with plots."""
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
        fig.savefig(os.path.join(output_dir, f"{col}_skew_{method}.png"))
        plt.close(fig)
        df[f"{col}_{method}"] = pd.Series(transformed, index=series.index)
    return df


def clustering_analysis(
    df: pd.DataFrame, target_col: str, output_dir: str, n_clusters: int = 3
) -> pd.DataFrame:
    """Run KMeans clustering and profile clusters against churn."""
    numeric = df.select_dtypes(include=np.number).drop(columns=[target_col], errors="ignore")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.fillna(0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    df["cluster"] = clusters
    if target_col in df.columns:
        profile = df.groupby("cluster")[target_col].mean()
        profile.to_frame("churn_rate").to_html(os.path.join(output_dir, "cluster_profile.html"))
        sns.barplot(x=profile.index, y=profile.values)
        plt.ylabel("Churn Rate")
        plt.savefig(os.path.join(output_dir, "cluster_churn_rate.png"))
        plt.close()
    return df


def pca_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Run PCA on numeric features and plot explained variance."""
    numeric = df.select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.fillna(0))
    pca = PCA()
    pca.fit(scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(range(1, len(explained) + 1), explained, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_explained_variance.png"))
    plt.close()


def woe_iv_analysis(
    df: pd.DataFrame, target_col: str, output_dir: str
) -> pd.DataFrame:
    """Compute WOE and IV for each feature and save tables."""
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
    woe_df = pd.DataFrame(woe_records)
    iv_df.to_html(os.path.join(output_dir, "iv_table.html"), index=False)
    woe_df.to_html(os.path.join(output_dir, "woe_table.html"), index=False)
    return iv_df


def random_forest_analysis(
    df: pd.DataFrame, target_col: str, output_dir: str
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """Train RandomForest, plot importances, and compute SHAP values."""
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_frame("importance").to_html(
        os.path.join(output_dir, "feature_importances.html")
    )
    plt.figure(figsize=(8, max(4, len(importances.head(20)) * 0.4)))
    sns.barplot(x=importances.head(20), y=importances.head(20).index)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rf_importances.png"))
    plt.close()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})
    perm_df.sort_values(by="importance", ascending=False).to_html(
        os.path.join(output_dir, "permutation_importance.html"), index=False
    )

    return model, X_test


def partial_dependence_analysis(
    model: RandomForestClassifier, X: pd.DataFrame, output_dir: str
) -> None:
    """Plot partial dependence for top two features and their interaction."""
    features = list(X.columns[:2])
    if len(features) >= 2:
        display = PartialDependenceDisplay.from_estimator(
            model, X, [features[0], features[1], (features[0], features[1])]
        )
        display.figure_.tight_layout()
        display.figure_.savefig(os.path.join(output_dir, "partial_dependence.png"))
        plt.close(display.figure_)


def survival_analysis(
    df: pd.DataFrame, target_col: str, tenure_col: str, output_dir: str
) -> None:
    """Generate Kaplan-Meier survival curve if tenure column exists."""
    if tenure_col in df.columns:
        kmf = KaplanMeierFitter()
        kmf.fit(df[tenure_col], event_observed=df[target_col])
        kmf.plot_survival_function()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "survival_curve.png"))
        plt.close()


def generate_summary(
    output_dir: str,
    missing_report: pd.DataFrame,
    high_corr: List[Tuple[str, str, float]],
    iv_df: pd.DataFrame,
) -> None:
    """Create a markdown summary of key insights."""
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
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Comprehensive churn analysis tool")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument(
        "--target-column", default="churn", help="Name of churn target column"
    )
    parser.add_argument(
        "--tenure-column", default="tenure", help="Name of tenure column if available"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory to save outputs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.csv_path)
    target_col = args.target_column
    if target_col not in df.columns:
        alt = [c for c in df.columns if c.lower() == target_col.lower()]
        if alt:
            target_col = alt[0]
        else:
            raise ValueError(f"Target column '{args.target_column}' not found in data")

    missing_report = report_missing_values(df)
    summary_statistics(df, args.output_dir)
    plot_distributions(df, args.output_dir)
    high_corr = correlation_analysis(df, args.output_dir)
    df = handle_skewness(df, args.output_dir)
    df = clustering_analysis(df, target_col, args.output_dir)
    pca_analysis(df, args.output_dir)
    iv_df = woe_iv_analysis(df, target_col, args.output_dir)
    model, X_test = random_forest_analysis(df.dropna(subset=[target_col]), target_col, args.output_dir)
    partial_dependence_analysis(model, X_test, args.output_dir)
    survival_analysis(df, target_col, args.tenure_column, args.output_dir)
    generate_summary(args.output_dir, missing_report, high_corr, iv_df)


if __name__ == "__main__":
    main()
