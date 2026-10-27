"""CineMind: Hit/Flop classification, box-office regression and movie recommender.

Run: python train.py   (writes models/, figures/, metrics.json)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
FIG, MOD = ROOT / "figures", ROOT / "models"
FIG.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", palette="deep")

# ------------------------------------------------------------------ load
raw = pd.read_csv(ROOT / "data" / "movie_metadata.csv")
raw["movie_title"] = raw["movie_title"].str.strip().str.replace("\xa0", "", regex=False)
raw = raw.drop_duplicates(subset=["movie_title", "title_year"]).reset_index(drop=True)
stats = {"raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1]),
         "missing_before": {k: int(v) for k, v in raw.isna().sum().items() if v}}

# ------------------------------------------------------------------ EDA (raw)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
raw.isna().sum().sort_values(ascending=False).head(12).plot.barh(ax=ax[0], color="#c0392b")
ax[0].set_title("Top missing-value counts"); ax[0].invert_yaxis()
sns.histplot(raw["imdb_score"].dropna(), bins=30, kde=True, ax=ax[1], color="#2c3e50")
ax[1].set_title("IMDb score distribution")
plt.tight_layout(); plt.savefig(FIG / "eda_missing_score.png", dpi=150); plt.close()

# ------------------------------------------------------------------ preprocessing
# Budget/gross are only comparable in one currency -> keep USA productions.
df = raw[raw["country"] == "USA"].dropna(subset=["gross", "budget", "title_year"]).copy()
df = df[(df["budget"] >= 1e5) & (df["gross"] >= 1e4)]          # drop unit-error rows
stats["usa_rows_with_money"] = int(len(df))
df["profit"] = df["gross"] - df["budget"]
df["roi"] = df["gross"] / df["budget"]
df["hit"] = (df["gross"] >= df["budget"]).astype(int)
df["log_gross"] = np.log10(df["gross"])
df["log_budget"] = np.log10(df["budget"])
df["decade"] = (df["title_year"] // 10 * 10).astype(int)
df["main_genre"] = df["genres"].str.split("|").str[0]
stats["hit_rate"] = round(float(df["hit"].mean()), 4)

# IQR outlier report for the numeric drivers
out = {}
for c in ["budget", "gross", "duration", "director_facebook_likes", "cast_total_facebook_likes"]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    out[c] = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
stats["iqr_outliers"] = out

# genre multi-hot (genres with >=60 films)
gdum = df["genres"].str.get_dummies("|")
GENRES = [g for g in gdum.columns if gdum[g].sum() >= 60]
for g in GENRES:
    df["g_" + g] = gdum[g]
df["is_english"] = (df["language"] == "English").astype(int)
df["log_budget_f"] = df["log_budget"]


def track_record(frame, key, prefix):
    """Mean log-ROI and film count of this person's *earlier* films only (no look-ahead leakage)."""
    frame = frame.sort_values("title_year")
    g = frame.groupby(key)
    lr = np.log10(frame["roi"])
    frame[prefix + "_prev_n"] = g.cumcount()
    frame[prefix + "_prev_logroi"] = (lr.groupby(frame[key]).cumsum() - lr) / frame[prefix + "_prev_n"].replace(0, np.nan)
    frame[prefix + "_prev_hit"] = (frame["hit"].groupby(frame[key]).cumsum() - frame["hit"]) / frame[prefix + "_prev_n"].replace(0, np.nan)
    return frame.sort_index()


df = track_record(df.dropna(subset=["director_name"]).combine_first(df), "director_name", "dir")
df = track_record(df.assign(actor_1_name=df["actor_1_name"].fillna("?")), "actor_1_name", "act1")
df["content_rating"] = df["content_rating"].replace(
    {"Not Rated": "Unrated", "Unrated": "Unrated", "Approved": "Unrated", "Passed": "Unrated",
     "GP": "PG", "M": "PG", "X": "NC-17"}).fillna("Unrated")
df.loc[~df["content_rating"].isin(["PG", "PG-13", "R", "G", "NC-17"]), "content_rating"] = "Unrated"

TRACK = ["dir_prev_n", "dir_prev_logroi", "dir_prev_hit", "act1_prev_n", "act1_prev_logroi", "act1_prev_hit"]
NUM = TRACK + ["log_budget_f", "budget", "duration", "title_year", "director_facebook_likes", "actor_1_facebook_likes",
       "actor_2_facebook_likes", "actor_3_facebook_likes", "cast_total_facebook_likes",
       "facenumber_in_poster", "aspect_ratio", "is_english"] + ["g_" + g for g in GENRES]
CAT = ["content_rating", "color"]
FEATURES = NUM + CAT
stats["features"] = FEATURES
stats["n_features_raw"] = len(FEATURES)

# ------------------------------------------------------------------ EDA (clean)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(df["log_budget"], bins=30, kde=True, ax=ax[0, 0], color="#2980b9")
ax[0, 0].set_title("log10(Budget) distribution")
sns.histplot(df["log_gross"], bins=30, kde=True, ax=ax[0, 1], color="#27ae60")
ax[0, 1].set_title("log10(Gross) distribution")
sns.boxplot(data=df[["duration"]], ax=ax[1, 0], color="#f39c12"); ax[1, 0].set_title("Duration outliers (box plot)")
sns.boxplot(x="content_rating", y="log_gross", data=df, ax=ax[1, 1],
            order=["G", "PG", "PG-13", "R", "NC-17", "Unrated"])
ax[1, 1].set_title("Gross by content rating")
plt.tight_layout(); plt.savefig(FIG / "eda_distributions.png", dpi=150); plt.close()

corr_cols = ["budget", "gross", "imdb_score", "duration", "title_year", "director_facebook_likes",
             "cast_total_facebook_likes", "num_voted_users", "num_critic_for_reviews", "movie_facebook_likes"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, annot_kws={"size": 8})
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(FIG / "eda_correlation.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
gs = pd.DataFrame({g: [df.loc[df["g_" + g] == 1, "hit"].mean(), int(df["g_" + g].sum())] for g in GENRES},
                  index=["hit_rate", "n"]).T.sort_values("hit_rate")
ax[0].barh(gs.index, gs["hit_rate"], color="#8e44ad"); ax[0].axvline(df["hit"].mean(), c="k", ls="--")
ax[0].set_title("Hit rate by genre (dashed = overall)")
yr = df.groupby("decade").agg(hit=("hit", "mean"), n=("hit", "size"))
ax[1].bar(yr.index.astype(str), yr["hit"], color="#16a085"); ax[1].set_title("Hit rate by decade")
plt.tight_layout(); plt.savefig(FIG / "eda_genre_decade.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="log_budget", y="log_gross", hue="hit", alpha=.5, palette={0: "#c0392b", 1: "#27ae60"})
lim = [df["log_budget"].min(), df["log_budget"].max()]; plt.plot(lim, lim, "k--", lw=1, label="break-even")
plt.title("Budget vs gross"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "eda_budget_gross.png", dpi=150); plt.close()

# ------------------------------------------------------------------ split + preprocessing pipeline
X, y_cls, y_reg = df[FEATURES], df["hit"], df["log_gross"]
Xtr, Xte, ytr, yte, rtr, rte = train_test_split(X, y_cls, y_reg, test_size=.2, random_state=SEED, stratify=y_cls)
stats["train_rows"], stats["test_rows"] = int(len(Xtr)), int(len(Xte))


def prep(scale=True):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(num), NUM),
                              ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                                ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])


# class balance is ~mildly skewed -> class_weight="balanced" instead of SMOTE
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf_models = {
    "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Decision Tree": Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED))]),
    "SVM (RBF)": Pipeline([("p", prep()), ("m", SVC(probability=True, class_weight="balanced", random_state=SEED))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("p", prep(False)), ("m", GradientBoostingClassifier(random_state=SEED))]),
}


def cls_scores(model):
    p = model.predict(Xte); pr = model.predict_proba(Xte)[:, 1]
    return {"accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p), "recall": recall_score(yte, p),
            "f1": f1_score(yte, p), "roc_auc": roc_auc_score(yte, pr)}


# ------------------------------------------------------------------ hyper-parameter tuning (RF + GB)
grids = {
    "Random Forest": {"m__n_estimators": [200, 400, 600], "m__max_depth": [6, 10, 14, None],
                      "m__min_samples_leaf": [1, 3, 5], "m__max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"m__n_estimators": [100, 200, 300], "m__learning_rate": [.03, .05, .1],
                          "m__max_depth": [2, 3, 4], "m__subsample": [.7, .85, 1.0]},
}
results_cls, fitted_cls, best_params = {}, {}, {}
for name, mdl in clf_models.items():
    cvs = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    if name in grids:
        s = RandomizedSearchCV(mdl, grids[name], n_iter=12, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
        s.fit(Xtr, ytr); mdl = s.best_estimator_; best_params[name] = {k: v for k, v in s.best_params_.items()}
        name_out = name + " (tuned)"
    else:
        mdl.fit(Xtr, ytr); name_out = name
    results_cls[name_out] = {**cls_scores(mdl), "cv_auc_mean": float(cvs.mean()), "cv_auc_std": float(cvs.std())}
    fitted_cls[name_out] = mdl
    print("CLS", name_out, {k: round(v, 3) for k, v in results_cls[name_out].items()})

champ_cls = max(results_cls, key=lambda k: results_cls[k]["roc_auc"])
best = fitted_cls[champ_cls]
stats["champion_classifier"], stats["best_params"] = champ_cls, best_params

# plots: ROC, confusion matrix, comparison
plt.figure(figsize=(6.5, 5.5))
for n, m in fitted_cls.items():
    f, t, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1]); plt.plot(f, t, label=f"{n} ({results_cls[n]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves – Hit/Flop"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "res_roc.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(best, Xte, yte, display_labels=["Flop", "Hit"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix – {champ_cls}"); plt.tight_layout(); plt.savefig(FIG / "res_confusion.png", dpi=150); plt.close()

cmp = pd.DataFrame(results_cls).T[["accuracy", "f1", "roc_auc"]]
cmp.plot.bar(figsize=(9, 4.5), rot=25); plt.ylim(.4, 1); plt.title("Classifier comparison (test set)")
plt.tight_layout(); plt.savefig(FIG / "res_model_comparison.png", dpi=150); plt.close()

# learning curve (train vs validation)
sizes, trs, vas = learning_curve(best, Xtr, ytr, cv=cv, scoring="accuracy", train_sizes=np.linspace(.15, 1, 7), n_jobs=-1)
plt.figure(figsize=(6.5, 4.5)); plt.plot(sizes, trs.mean(1), "o-", label="Training"); plt.plot(sizes, vas.mean(1), "o-", label="Validation")
plt.xlabel("Training examples"); plt.ylabel("Accuracy"); plt.title(f"Learning curve – {champ_cls}"); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "res_learning_curve.png", dpi=150); plt.close()

# feature importance (from the tree-based champion) via permutation on the raw columns
from sklearn.inspection import permutation_importance
pi = permutation_importance(best, Xte, yte, n_repeats=8, random_state=SEED, scoring="roc_auc", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values().tail(12)
plt.figure(figsize=(7, 5)); imp.plot.barh(color="#e67e22"); plt.title("Permutation importance (ROC-AUC drop)")
plt.tight_layout(); plt.savefig(FIG / "res_importance.png", dpi=150); plt.close()
stats["top_features"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()}

# ------------------------------------------------------------------ regression (log10 gross)
reg_models = {
    "Linear Regression": Pipeline([("p", prep()), ("m", LinearRegression())]),
    "Ridge": Pipeline([("p", prep()), ("m", Ridge(alpha=10))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("p", prep(False)), ("m", GradientBoostingRegressor(random_state=SEED))]),
}
reg_grid = {"m__n_estimators": [150, 250, 400], "m__learning_rate": [.03, .05, .1], "m__max_depth": [2, 3, 4], "m__subsample": [.7, .85, 1.0]}
results_reg, fitted_reg = {}, {}
for name, mdl in reg_models.items():
    if name == "Gradient Boosting":
        s = RandomizedSearchCV(mdl, reg_grid, n_iter=12, scoring="r2", cv=5, random_state=SEED, n_jobs=-1).fit(Xtr, rtr)
        mdl, name = s.best_estimator_, name + " (tuned)"; best_params["GB Regressor"] = s.best_params_
    else:
        mdl.fit(Xtr, rtr)
    p = mdl.predict(Xte)
    usd_rmse = float(np.sqrt(mean_squared_error(10 ** rte, 10 ** p)))
    results_reg[name] = {"r2": r2_score(rte, p), "rmse_log": float(np.sqrt(mean_squared_error(rte, p))),
                         "mae_log": mean_absolute_error(rte, p), "rmse_usd_m": usd_rmse / 1e6}
    fitted_reg[name] = mdl
    print("REG", name, {k: round(v, 3) for k, v in results_reg[name].items()})
champ_reg = max(results_reg, key=lambda k: results_reg[k]["r2"])
stats["champion_regressor"] = champ_reg
pr = fitted_reg[champ_reg].predict(Xte)
plt.figure(figsize=(6, 5.5)); plt.scatter(rte, pr, alpha=.4, s=14); lim = [rte.min(), rte.max()]
plt.plot(lim, lim, "r--"); plt.xlabel("Actual log10(gross)"); plt.ylabel("Predicted log10(gross)")
plt.title(f"Actual vs predicted – {champ_reg}"); plt.tight_layout(); plt.savefig(FIG / "res_actual_vs_pred.png", dpi=150); plt.close()
pd.DataFrame(results_reg).T[["r2"]].plot.bar(figsize=(6.5, 4), rot=20, legend=False, color="#2ecc71")
plt.title("Regression R² (test set)"); plt.tight_layout(); plt.savefig(FIG / "res_reg_comparison.png", dpi=150); plt.close()

# ------------------------------------------------------------------ recommender (content-based, all films)
rec = raw.dropna(subset=["movie_title"]).copy()
for c in ["genres", "plot_keywords", "director_name", "actor_1_name", "actor_2_name", "actor_3_name"]:
    rec[c] = rec[c].fillna("")


def soup(r):
    ppl = [str(r[c]).replace(" ", "") for c in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name"] if r[c]]
    return " ".join(r["genres"].split("|") * 2 + r["plot_keywords"].replace(" ", "_").split("|") + ppl * 2)


rec["soup"] = rec.apply(soup, axis=1)
tf = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2).fit(rec["soup"])
rec_cols = ["movie_title", "title_year", "genres", "director_name", "imdb_score", "gross", "budget"]
joblib.dump({"tfidf": tf, "matrix": tf.transform(rec["soup"]), "meta": rec[rec_cols].reset_index(drop=True)}, MOD / "recommender.joblib")
stats["recommender_titles"] = int(len(rec))

# ------------------------------------------------------------------ persist
joblib.dump({"clf": best, "reg": fitted_reg[champ_reg], "features": FEATURES, "num": NUM, "cat": CAT, "genres": GENRES}, MOD / "models.joblib")
df.to_csv(ROOT / "data" / "movies_clean.csv", index=False)
json.dump({"stats": stats, "classification": results_cls, "regression": results_reg}, open(ROOT / "metrics.json", "w"), indent=2, default=str)
print(json.dumps({"champ_cls": champ_cls, "champ_reg": champ_reg, "hit_rate": stats["hit_rate"], "rows": stats["usa_rows_with_money"]}))
