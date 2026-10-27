"""CineMind Streamlit app. Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).parent
st.set_page_config(page_title="CineMind", page_icon="🎬", layout="wide")


@st.cache_resource
def load():
    m = joblib.load(ROOT / "models" / "models.joblib")
    r = joblib.load(ROOT / "models" / "recommender.joblib")
    d = pd.read_csv(ROOT / "data" / "movies_clean.csv")
    met = json.load(open(ROOT / "metrics.json"))
    return m, r, d, met


M, R, D, MET = load()
GENRES = M["genres"]


def person_record(name, col, prefix):
    """Career stats of a known person from the dataset (neutral NaN for newcomers)."""
    h = D[D[col] == name]
    if not name or h.empty:
        return {f"{prefix}_prev_n": 0, f"{prefix}_prev_logroi": np.nan, f"{prefix}_prev_hit": np.nan}, 0
    return {f"{prefix}_prev_n": len(h), f"{prefix}_prev_logroi": float(np.log10(h["roi"]).mean()),
            f"{prefix}_prev_hit": float(h["hit"].mean())}, len(h)


st.title("🎬 CineMind")
st.caption("Box-office & hit/flop prediction and movie recommendation · ML mini project")
t1, t2, t3, t4 = st.tabs(["🎯 Predict a film", "🍿 Recommend", "📊 Industry insights", "🧪 Model report"])

# ----------------------------------------------------------------- predict
with t1:
    st.subheader("Pre-release prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        budget = st.number_input("Budget (USD millions)", 0.1, 400.0, 60.0, 5.0)
        duration = st.slider("Duration (min)", 60, 220, 115)
        year = st.slider("Release year", 1980, 2026, 2024)
        rating = st.selectbox("Content rating", ["PG-13", "R", "PG", "G", "NC-17", "Unrated"])
    with c2:
        genres = st.multiselect("Genres", GENRES, default=["Action", "Adventure"])
        english = st.checkbox("English language", True)
        color = st.selectbox("Colour", ["Color", "Black and White"])
        faces = st.slider("Faces on poster", 0, 10, 1)
    with c3:
        dirs = sorted(D["director_name"].dropna().unique())
        acts = sorted(D["actor_1_name"].dropna().unique())
        dname = st.selectbox("Director (type to search)", [""] + dirs, format_func=lambda x: x or "— new / unknown —")
        aname = st.selectbox("Lead actor (type to search)", [""] + acts, format_func=lambda x: x or "— new / unknown —")
        dlikes = st.number_input("Director Facebook likes", 0, 500000, int(D.loc[D.director_name == dname, "director_facebook_likes"].median() if dname else 500))
        a1 = st.number_input("Lead actor Facebook likes", 0, 700000, int(D.loc[D.actor_1_name == aname, "actor_1_facebook_likes"].median() if aname else 2000))
        a2 = st.number_input("Actor 2 Facebook likes", 0, 700000, 1000)
        a3 = st.number_input("Actor 3 Facebook likes", 0, 700000, 500)

    if st.button("Predict", type="primary"):
        row = {"budget": budget * 1e6, "log_budget_f": np.log10(budget * 1e6), "duration": duration, "title_year": year,
               "director_facebook_likes": dlikes, "actor_1_facebook_likes": a1, "actor_2_facebook_likes": a2,
               "actor_3_facebook_likes": a3, "cast_total_facebook_likes": a1 + a2 + a3, "facenumber_in_poster": faces,
               "aspect_ratio": 2.35, "is_english": int(english), "content_rating": rating, "color": color}
        dr, dn = person_record(dname, "director_name", "dir")
        ar, an = person_record(aname, "actor_1_name", "act1")
        row.update(dr)
        row.update(ar)
        row.update({"g_" + g: int(g in genres) for g in GENRES})
        X = pd.DataFrame([row])[M["features"]]
        p_hit = float(M["clf"].predict_proba(X)[0, 1])
        gross = float(10 ** M["reg"].predict(X)[0])
        k1, k2, k3 = st.columns(3)
        k1.metric("Hit probability", f"{p_hit:.0%}", "Likely HIT" if p_hit >= .5 else "Likely FLOP", delta_color="normal" if p_hit >= .5 else "inverse")
        k2.metric("Predicted gross", f"${gross / 1e6:,.1f}M")
        k3.metric("Predicted profit", f"${(gross - budget * 1e6) / 1e6:,.1f}M")
        st.progress(min(max(p_hit, 0.0), 1.0))
        st.caption(f"Director history used: {dn} film(s) · Lead-actor history used: {an} film(s). "
                   "Typical error on gross is about ±0.4 in log10 units (roughly 2.5×); treat as an estimate, not a forecast.")

# ----------------------------------------------------------------- recommend
with t2:
    st.subheader("Content-based recommender")
    meta = R["meta"]
    title = st.selectbox("Pick a film you like", meta["movie_title"].tolist(), index=int(meta.index[meta.movie_title == "Avatar"][0]) if (meta.movie_title == "Avatar").any() else 0)
    n = st.slider("How many?", 3, 15, 8)
    i = meta.index[meta.movie_title == title][0]
    sims = linear_kernel(R["matrix"][i], R["matrix"]).ravel()
    sims[i] = -1
    top = np.argsort(-sims)[:n]
    out = meta.iloc[top].copy()
    out["similarity"] = sims[top].round(3)
    out["gross"] = (out["gross"] / 1e6).round(1)
    st.dataframe(out.rename(columns={"movie_title": "Title", "title_year": "Year", "genres": "Genres", "director_name": "Director",
                                     "imdb_score": "IMDb", "gross": "Gross ($M)", "similarity": "Similarity"})
                 [["Title", "Year", "Genres", "Director", "IMDb", "Gross ($M)", "Similarity"]], hide_index=True, width='stretch')

# ----------------------------------------------------------------- insights
with t3:
    st.subheader("What drives box-office success?")
    f1, f2 = st.columns(2)
    lo, hi = f1.slider("Years", int(D.title_year.min()), int(D.title_year.max()), (1990, 2016))
    gsel = f2.multiselect("Genre filter", GENRES)
    S = D[(D.title_year >= lo) & (D.title_year <= hi)]
    for g in gsel:
        S = S[S["g_" + g] == 1]
    a, b, c, d = st.columns(4)
    a.metric("Films", len(S)); b.metric("Hit rate", f"{S.hit.mean():.0%}")
    c.metric("Median budget", f"${S.budget.median() / 1e6:.0f}M"); d.metric("Median ROI", f"{S.roi.median():.2f}×")
    st.markdown("**Hit rate by decade**")
    st.bar_chart(S.groupby("decade").hit.mean())
    st.markdown("**Budget vs gross (log scale)**")
    st.scatter_chart(S, x="log_budget", y="log_gross", color="hit", height=360)
    st.markdown("**Top directors by average ROI (≥3 films)**")
    dd = S.groupby("director_name").agg(films=("roi", "size"), median_roi=("roi", "median"), avg_gross_m=("gross", lambda x: x.mean() / 1e6)).query("films>=3").sort_values("median_roi", ascending=False).head(10).round(2)
    st.dataframe(dd, width='stretch')

# ----------------------------------------------------------------- report
with t4:
    st.subheader("Model performance (held-out 20% test set)")
    cl = pd.DataFrame(MET["classification"]).T.round(3)
    st.markdown(f"**Hit/Flop classifiers** — champion: `{MET['stats']['champion_classifier']}`")
    st.dataframe(cl[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean"]], width='stretch')
    rg = pd.DataFrame(MET["regression"]).T.round(3)
    st.markdown(f"**Gross regressors (log10 target)** — champion: `{MET['stats']['champion_regressor']}`")
    st.dataframe(rg[["r2", "rmse_log", "mae_log"]], width='stretch')
    g1, g2 = st.columns(2)
    g1.image(str(ROOT / "figures" / "res_roc.png")); g2.image(str(ROOT / "figures" / "res_importance.png"))
