"""
Fake & Clone Twitter Account Detection - end-to-end analysis pipeline.

Stages (each maps to a Business Analyst activity):
  1. Load data            -> data collection
  2. Clean data           -> data quality checks
  3. EDA + charts         -> understanding patterns
  4. Feature engineering  -> turning raw fields into measurable signals
  5. KPI summary          -> business metrics for stakeholders
  6. Classification       -> fake (bot) vs genuine, 4 algorithms compared
  7. Clone detection      -> Euclidean distance between account profiles
  8. Export               -> CSVs ready for Power BI / Tableau / Excel dashboards

Usage:
    python src/pipeline.py                         # uses data/twitter_bot_detection.csv, else data/sample_accounts.csv
    python src/pipeline.py --data data/yourfile.csv
"""
import argparse
import difflib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ---- Column names in the Kaggle file. Change here if your file differs. ----
COL = {
    "id": "User ID",
    "username": "Username",
    "tweet": "Tweet",
    "retweets": "Retweet Count",
    "mentions": "Mention Count",
    "followers": "Follower Count",
    "verified": "Verified",
    "label": "Bot Label",
    "location": "Location",
    "created": "Created At",
    "hashtags": "Hashtags",
}
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ 1. LOAD
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [v for v in COL.values() if v not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in file: {missing}\nActual columns: {list(df.columns)}\n"
                         "Edit the COL dictionary at the top of src/pipeline.py.")
    print(f"[1] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {path.name}")
    return df


# ----------------------------------------------------------------- 2. CLEAN
def clean(df: pd.DataFrame) -> pd.DataFrame:
    report = {"rows_before": len(df),
              "missing_values": df.isna().sum()[df.isna().sum() > 0].to_dict(),
              "duplicate_rows": int(df.duplicated().sum())}
    df = df.drop_duplicates().copy()
    df[COL["location"]] = df[COL["location"]].fillna("Unknown")
    df[COL["hashtags"]] = df[COL["hashtags"]].fillna("")
    df[COL["tweet"]] = df[COL["tweet"]].fillna("")
    df[COL["verified"]] = df[COL["verified"]].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    df[COL["created"]] = pd.to_datetime(df[COL["created"]], errors="coerce")
    df = df.dropna(subset=[COL["label"], COL["username"]])
    df[COL["label"]] = df[COL["label"]].astype(int)
    report["rows_after"] = len(df)
    (OUT / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[2] Cleaned data: {report['rows_before']:,} -> {report['rows_after']:,} rows "
          f"(duplicates removed: {report['duplicate_rows']})")
    return df.reset_index(drop=True)


# ---------------------------------------------------------- 4. FEATURE ENG.
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    name = df[COL["username"]].astype(str)
    df["username_length"] = name.str.len()
    df["username_digit_count"] = name.str.count(r"\d")
    df["username_digit_ratio"] = df["username_digit_count"] / df["username_length"]
    df["tweet_length"] = df[COL["tweet"]].astype(str).str.len()
    df["hashtag_count"] = df[COL["hashtags"]].astype(str).str.count("#")
    ref = df[COL["created"]].max()
    df["account_age_days"] = (ref - df[COL["created"]]).dt.days.fillna(0)
    df["engagement_per_follower"] = (df[COL["retweets"]] + df[COL["mentions"]]) / (df[COL["followers"]] + 1)
    return df


FEATURES = ["username_length", "username_digit_count", "username_digit_ratio", "tweet_length",
            "hashtag_count", "account_age_days", "engagement_per_follower",
            "Retweet Count", "Mention Count", "Follower Count", "Verified"]


# ------------------------------------------------------------------- 3. EDA
def eda_charts(df: pd.DataFrame) -> None:
    lab = COL["label"]
    df = df.assign(Type=df[lab].map({0: "Genuine", 1: "Fake/Bot"}))
    pal = {"Genuine": "#2a7ab0", "Fake/Bot": "#d1495b"}

    fig, ax = plt.subplots(figsize=(5, 4))
    df["Type"].value_counts().plot.bar(ax=ax, color=[pal[i] for i in df["Type"].value_counts().index])
    ax.set_title("Class balance: genuine vs fake accounts"); ax.set_ylabel("Accounts"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(CHARTS / "01_class_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (c, t) in zip(axes.ravel(), [(COL["followers"], "Follower count (log)"),
                                         (COL["retweets"], "Retweet count"),
                                         ("username_digit_count", "Digits in username"),
                                         ("account_age_days", "Account age (days)")]):
        sns.histplot(data=df, x=c, hue="Type", palette=pal, bins=40, ax=ax, log_scale=(c == COL["followers"]))
        ax.set_title(t)
    fig.tight_layout(); fig.savefig(CHARTS / "02_distributions.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[FEATURES + [lab]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap"); fig.tight_layout(); fig.savefig(CHARTS / "03_correlation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (df.groupby(COL["location"])[lab].mean().sort_values(ascending=False) * 100).plot.bar(ax=ax, color="#d1495b")
    ax.set_title("Fake-account rate by location (%)"); ax.set_ylabel("%"); fig.tight_layout()
    fig.savefig(CHARTS / "04_fake_rate_by_location.png", dpi=150); plt.close(fig)
    print("[3] EDA charts saved to outputs/charts/")


# ------------------------------------------------------------------ 5. KPIs
def kpis(df: pd.DataFrame) -> pd.DataFrame:
    lab = COL["label"]
    fake = df[df[lab] == 1]; real = df[df[lab] == 0]
    rows = [
        ("Total accounts analysed", len(df)),
        ("Fake/bot accounts", len(fake)),
        ("Fake account rate (%)", round(100 * len(fake) / len(df), 2)),
        ("Avg followers - fake", round(fake[COL["followers"]].mean(), 1)),
        ("Avg followers - genuine", round(real[COL["followers"]].mean(), 1)),
        ("Avg retweets - fake", round(fake[COL["retweets"]].mean(), 1)),
        ("Avg retweets - genuine", round(real[COL["retweets"]].mean(), 1)),
        ("Avg digits in username - fake", round(fake["username_digit_count"].mean(), 2)),
        ("Avg digits in username - genuine", round(real["username_digit_count"].mean(), 2)),
        ("Verified share among fake (%)", round(100 * fake[COL["verified"]].mean(), 2)),
    ]
    k = pd.DataFrame(rows, columns=["KPI", "Value"])
    k.to_csv(OUT / "kpi_summary.csv", index=False)
    print("[5] KPI summary:\n" + k.to_string(index=False))
    return k


# ------------------------------------------------------- 6. CLASSIFICATION
def classify(df: pd.DataFrame):
    X, y = df[FEATURES], df[COL["label"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (Euclidean)": KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    results, fitted = [], {}
    for n, m in models.items():
        m.fit(Xtr_s, ytr)
        p = m.predict(Xte_s); pr = m.predict_proba(Xte_s)[:, 1]
        results.append({"Model": n, "Accuracy": accuracy_score(yte, p), "Precision": precision_score(yte, p),
                        "Recall": recall_score(yte, p), "F1": f1_score(yte, p), "ROC_AUC": roc_auc_score(yte, pr)})
        fitted[n] = (m, p)
    res = pd.DataFrame(results).round(4).sort_values("F1", ascending=False)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    print("[6] Model comparison:\n" + res.to_string(index=False))

    best = res.iloc[0]["Model"]
    cm = confusion_matrix(yte, fitted[best][1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Genuine", "Fake"], yticklabels=["Genuine", "Fake"])
    ax.set_title(f"Confusion matrix - {best}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(CHARTS / "05_confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp = pd.Series(fitted["Random Forest"][0].feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(ax=ax, color="#2a7ab0"); ax.set_title("What signals matter most? (Random Forest)")
    fig.tight_layout(); fig.savefig(CHARTS / "06_feature_importance.png", dpi=150); plt.close(fig)
    imp.sort_values(ascending=False).rename("importance").to_csv(OUT / "feature_importance.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(ax=ax)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison"); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(CHARTS / "07_model_comparison.png", dpi=150); plt.close(fig)
    return res, best


# ----------------------------------------------------- 7. CLONE DETECTION
def detect_clones(df: pd.DataFrame, dist_pct: float = 2.0, name_sim: float = 0.80) -> pd.DataFrame:
    """Two accounts are a suspected clone pair if
       (a) their standardised profile vectors are very close (Euclidean distance in the lowest
           `dist_pct` percent of all nearest-neighbour distances) AND
       (b) their usernames are textually similar (difflib ratio >= name_sim)."""
    cols = [COL["followers"], COL["retweets"], COL["mentions"], "account_age_days", "tweet_length", "hashtag_count"]
    Z = StandardScaler().fit_transform(np.log1p(df[cols].clip(lower=0)))
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(Z)
    dist, idx = nn.kneighbors(Z)
    d, j = dist[:, 1], idx[:, 1]
    thr = np.percentile(d, dist_pct)
    names = df[COL["username"]].astype(str).values
    pairs = []
    for i in np.where(d <= thr)[0]:
        sim = difflib.SequenceMatcher(None, names[i], names[j[i]]).ratio()
        if sim >= name_sim and i < j[i]:
            pairs.append({"account_A": names[i], "account_B": names[j[i]], "euclidean_distance": round(float(d[i]), 4),
                          "username_similarity": round(sim, 3),
                          "label_A": int(df[COL["label"]].iloc[i]), "label_B": int(df[COL["label"]].iloc[j[i]])})
    out = pd.DataFrame(pairs).sort_values("euclidean_distance") if pairs else pd.DataFrame(
        columns=["account_A", "account_B", "euclidean_distance", "username_similarity", "label_A", "label_B"])
    out.to_csv(OUT / "suspected_clone_pairs.csv", index=False)
    print(f"[7] Suspected clone pairs found: {len(out)} (distance threshold = {thr:.3f})")
    return out


# ----------------------------------------------------------------- MAIN
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    path = Path(a.data) if a.data else next(
        (p for p in [ROOT / "data" / "twitter_bot_detection.csv", ROOT / "data" / "sample_accounts.csv"] if p.exists()), None)
    if path is None:
        raise SystemExit("No data file found. Put the Kaggle CSV in data/ (see README) "
                         "or run: python src/generate_sample_data.py")
    df = add_features(clean(load(path)))
    eda_charts(df)
    kpis(df)
    res, best = classify(df)
    detect_clones(df)
    df.to_csv(OUT / "dashboard_ready_accounts.csv", index=False)
    print(f"[8] Done. Best model: {best}. All files are in outputs/")


if __name__ == "__main__":
    main()
