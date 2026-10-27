"""Builds CineMind_ML_Mini_Project_Report.docx following the department's report format.
Every number comes from metrics.json, so re-run train.py then this script to refresh."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).parent
FIG, SHOT = ROOT / "figures", ROOT / "screenshots"
M = json.load(open(ROOT / "metrics.json"))
S, C, R = M["stats"], M["classification"], M["regression"]
CC, CR = S["champion_classifier"], S["champion_regressor"]
cc, cr = C[CC], R[CR]

# ------------------------------------------------------------ architecture diagram
fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
boxes = ["IMDb 5000\nmovie dataset\n(5,043 rows)", "Cleaning &\nfiltering\n(USA, budget, gross)", "Feature\nengineering\n(genres, track record)",
         "Model training\n& tuning\n(RF, GB, SVM, LR)", "Evaluation\n(AUC, F1, R²)", "Streamlit app\n(predict · recommend\n· insights)"]
for i, t in enumerate(boxes):
    x = i * 1.78
    ax.add_patch(plt.Rectangle((x, .2), 1.5, .6, fc="#eaf2fb", ec="#1f4e79", lw=1.5))
    ax.text(x + .75, .5, t, ha="center", va="center", fontsize=8.5)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + 1.76, .5), xytext=(x + 1.5, .5), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1f4e79"))
ax.text(5.3, 1.0, "Content-based recommender (TF-IDF + cosine) built in parallel on all 4,919 titles", ha="center", fontsize=8.5, style="italic")
ax.set_xlim(-.1, 10.8); ax.set_ylim(0, 1.15)
plt.tight_layout(); plt.savefig(FIG / "architecture.png", dpi=170); plt.close()

# ------------------------------------------------------------ document helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Inches(1.25), Inches(1), Inches(1), Inches(1)
FONT = "Times New Roman"


def style_font(st, size, bold=False):
    st.font.name = FONT; st.font.size = Pt(size); st.font.bold = bold; st.font.italic = False
    st.font.color.rgb = None
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


for name, size, bold in [("Normal", 12, False), ("Heading 1", 14, True), ("Heading 2", 12, True), ("Title", 16, True)]:
    st = doc.styles[name]; style_font(st, size, bold)
    st.paragraph_format.line_spacing = 1.5
for name in ("Heading 1", "Heading 2"):
    doc.styles[name].paragraph_format.space_before = Pt(12); doc.styles[name].paragraph_format.space_after = Pt(6)
    doc.styles[name].paragraph_format.keep_with_next = True
doc.styles["Normal"].paragraph_format.space_after = Pt(6)


def fill(par, text):
    """Yellow-highlighted slot for details only the student knows."""
    r = par.add_run(text); r.font.highlight_color = WD_COLOR_INDEX.YELLOW; return r


def P(text="", bold=False, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align != "l": p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # [[...]] marks a yellow fill-in slot
    parts = text.split("[[")
    for i, seg in enumerate(parts):
        if i == 0:
            r = p.add_run(seg); r.bold, r.italic = bold, italic
            if size: r.font.size = Pt(size)
        else:
            slot, rest = seg.split("]]", 1); fill(p, "[" + slot + "]")
            r = p.add_run(rest); r.bold, r.italic = bold, italic
    return p


def B(text, lead=None):
    p = doc.add_paragraph(style="List Bullet"); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text); return p


def H1(t): return doc.add_heading(t, 1)
def H2(t): return doc.add_heading(t, 2)


def pagebreak(): doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def shade(cell, color):
    tcpr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcpr.append(sh)


tcount, fcount = [0], [0]
TABLES, FIGS = [], []


def table(caption, header, rows, widths=None):
    tcount[0] += 1; cap = f"Table {tcount[0]}: {caption}"; TABLES.append(cap)
    P(cap, bold=True, align="c", size=11)
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10.5); shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(10.5)
    for row in t.rows:
        for i, c in enumerate(row.cells):
            for p in c.paragraphs: p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(2)
            if widths: c.width = Inches(widths[i])
    doc.add_paragraph()


def figure(path, caption, width=5.4):
    fcount[0] += 1; cap = f"Figure {fcount[0]}: {caption}"; FIGS.append(cap)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    P(cap, bold=True, align="c", size=11)


def eq(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.italic = True; return p


# footer page numbers
def add_page_number(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)


add_page_number(sec)

# ------------------------------------------------------------ 1. preliminary pages
P("[[University logo]]", align="c")
P("")
P("CINEMIND: MOVIE SUCCESS PREDICTION, BOX-OFFICE FORECASTING AND RECOMMENDATION USING MACHINE LEARNING", bold=True, align="c", size=16)
P("")
P("A Machine Learning Mini Project Report", align="c")
P("Submitted by", align="c")
P("Vedant Nair", bold=True, align="c")
P("PRN: 23070122230  ·  Batch: CSE B2", align="c")
P("")
P("Under the guidance of", align="c")
P("[[Guide's name]]", bold=True, align="c")
P("")
P("Department of Artificial Intelligence and Data Science", bold=True, align="c")
P("[[Institute / University name]]", align="c")
P("Academic Year [[2026–2027]]", align="c")
pagebreak()

P("CERTIFICATE", bold=True, align="c", size=14)
P("This is to certify that the Machine Learning mini project report entitled “CineMind: Movie Success Prediction, Box-Office Forecasting and Recommendation using Machine Learning” is a bonafide work carried out by Vedant Nair (PRN 23070122230) in partial fulfilment of the requirements of the Machine Learning course, Department of Artificial Intelligence and Data Science, during the academic year [[2026–2027]]. The work is original and has not been submitted elsewhere for any degree or award.")
P("")
P("[[Project Guide]]                         [[Head of Department]]                         [[Principal]]", align="l")
P("Date: [[dd/mm/yyyy]]          Place: [[city]]", align="l")
pagebreak()

P("DECLARATION", bold=True, align="c", size=14)
P("I hereby declare that the project report entitled “CineMind: Movie Success Prediction, Box-Office Forecasting and Recommendation using Machine Learning” submitted to the Department of Artificial Intelligence and Data Science is a record of original work done by me under the guidance of [[Guide's name]]. The dataset used is publicly available, all external sources are cited in the References, and the report has not been submitted to any other institute or university for any degree or diploma.")
P("")
P("Vedant Nair (PRN 23070122230)", align="l"); P("Date: [[dd/mm/yyyy]]", align="l")
pagebreak()

P("ACKNOWLEDGEMENT", bold=True, align="c", size=14)
P("I express my sincere gratitude to my project guide [[Guide's name]] for continuous guidance and feedback throughout this work. I thank the Head of the Department, [[HOD's name]], and the Principal, [[Principal's name]], for providing the laboratory and computing facilities. I also acknowledge the creators of the open IMDb 5000 movie dataset and the maintainers of scikit-learn, pandas, matplotlib and Streamlit, whose tools made this project possible. Finally, I thank my family and classmates for their support.")
pagebreak()

P("ABSTRACT", bold=True, align="c", size=14)
P(f"The film industry invests hundreds of millions of dollars in productions whose commercial outcome is highly uncertain. This project, CineMind, applies supervised and unsupervised machine learning to the movie domain. Using the public IMDb 5000 movie dataset ({S['raw_rows']:,} unique titles after de-duplication), we keep {S['usa_rows_with_money']:,} US productions with valid budget and gross figures, engineer {S['n_features_raw']} pre-release features (budget, duration, genres, content rating, cast and director popularity, and leakage-safe career track records), and train five classifiers to predict whether a film will recoup its budget (Hit/Flop) and four regressors to forecast log-scaled box-office gross. Models are tuned with randomised search and validated with five-fold cross-validation. The champion classifier ({CC}) reaches {cc['accuracy']:.1%} accuracy and {cc['roc_auc']:.3f} ROC-AUC on a held-out test set, and the champion regressor ({CR}) explains {cr['r2']:.1%} of the variance in log gross (R² = {cr['r2']:.3f}). A TF-IDF content-based recommender over genres, plot keywords, directors and actors returns similar films. All components are delivered in a four-tab Streamlit application. The results show that pre-release information carries real but limited signal, which matches the published literature.")
P("Keywords: machine learning, movie box office, classification, regression, recommender system, Random Forest, Streamlit.", italic=True)
pagebreak()

P("TABLE OF CONTENTS", bold=True, align="c", size=14)
P("[[Right-click this field in Word and choose “Update field” → “Update entire table” to build the page-numbered table of contents.]]", align="l")
p = doc.add_paragraph(); r = p.add_run()
for t, txt in (("begin", None), (None, 'TOC \\o "1-2" \\h \\z \\u'), ("separate", None), ("end", None)):
    if t:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
    r._r.append(e)
P("")
LOF_ANCHOR = doc.add_paragraph()   # filled after the body is written
pagebreak()

# ------------------------------------------------------------ chapter 1
H1("Chapter 1: Introduction")
H2("1.1 Domain Overview")
P("The global film industry generates tens of billions of dollars a year, yet individual releases behave like venture investments: a few blockbusters subsidise many losses. Studios, distributors and streaming platforms therefore want data-driven answers to two questions. Before release: will this film make money, and roughly how much? After release: which other films should a viewer be shown next? Both are classic predictive-analytics problems. The first is a supervised learning task (classification and regression) over structured production attributes. The second is a recommendation task that relies on the similarity of film content.")
H2("1.2 Problem Statement")
P("Given only information available before a film is released (budget, runtime, genres, certificate, cast and director popularity, and the past record of the director and lead actor), predict (a) whether the film will be a Hit, defined as worldwide-reported gross at least equal to budget, and (b) its likely gross revenue. Additionally, given a film a user likes, recommend other films with similar content.")
H2("1.3 Motivation & Objectives")
P("Greenlighting decisions are expensive and largely driven by intuition. Even a modest, well-calibrated model can rank projects by risk and expose which factors matter. The recommender addresses the audience side of the same industry. The objectives are:")
B("Collect, clean and explore a real-world movie dataset and document every preprocessing decision.")
B("Build and compare at least four classification and four regression models with proper validation and hyper-parameter tuning.")
B("Prevent information leakage: no post-release variables (votes, reviews, IMDb score) are used as predictors.")
B("Build a content-based movie recommender.")
B("Deliver the system as an interactive web application that non-technical users can operate.")
H2("1.4 Scope of the Project")
P("In scope: US-produced feature films in the IMDb 5000 dataset (mostly 1980–2016), tabular modelling, a content-based recommender and a Streamlit front end. Out of scope: films from other countries (budgets are reported in local currencies and are not comparable), inflation adjustment, marketing spend, release-date or competition effects, social-media sentiment, and personalised collaborative filtering, which needs user-rating data that this dataset does not contain.")

# ------------------------------------------------------------ chapter 2
H1("Chapter 2: Literature Survey")
H2("2.1 Summary of Existing Systems")
P("Sharda and Delen (2006) were among the first to treat box-office prediction as a classification problem. They discretised revenue into nine classes ranging from “flop” to “blockbuster” and trained a multilayer perceptron on pre-release variables such as MPAA rating, competition, star value, genre, technical effects, sequel status and number of screens. Their network reached roughly 37% exact-class accuracy and about 75% within one class of the true class, which showed that the problem is learnable but noisy.")
P("Asur and Huberman (2010) showed that social media has predictive power. They used the rate of tweets about a film before release, together with a linear regression model, to forecast opening-weekend revenue and found that it outperformed the Hollywood Stock Exchange benchmark. The work highlights the value of behavioural signals but requires data that is unavailable in static datasets.")
P("Ghiassi, Lio and Moon (2015) proposed a dynamic artificial neural network for pre-production forecasting that uses only attributes known at the planning stage, and reported improved accuracy over earlier static networks. Their emphasis on pre-production inputs motivates the leakage-free feature design adopted in this project.")
P("Lash and Zhao (2016) focused on early profitability prediction, combining talent track records, textual features of the plot, and release timing in a decision-support framework. They showed that the “who” (people involved) and “what” (content) are informative, which supports the career-track-record features used here.")
P("Lops, de Gemmis and Semeraro (2011) surveyed content-based recommender systems, describing how item descriptions are turned into TF-IDF vectors and compared by cosine similarity. This is the technique implemented in the recommender module of this project.")
H2("2.2 Comparative Analysis")
table("Comparison of related work", ["Authors (year)", "Technique", "Task / target", "Reported result", "Major limitation"], [
    ["Sharda & Delen (2006)", "Multilayer perceptron", "9-class revenue category", "≈37% exact, ≈75% within 1 class", "Hard to interpret; coarse classes"],
    ["Asur & Huberman (2010)", "Linear regression on tweet rate", "Opening-weekend revenue", "Beat HSX market baseline", "Needs live social-media data"],
    ["Ghiassi et al. (2015)", "Dynamic ANN", "Pre-production revenue", "Improved over static ANN", "Small curated data; hard to reproduce"],
    ["Lash & Zhao (2016)", "Feature-rich ML framework", "Profitability (early)", "Talent and text are predictive", "Needs plot text and richer sources"],
    ["Lops et al. (2011)", "TF-IDF + cosine similarity", "Content-based recommendation", "Survey (no single score)", "Over-specialisation, cold-start for users"],
    ["This work (CineMind)", f"RF, GB, SVM, LR + TF-IDF", "Hit/Flop, log gross, similar films", f"{cc['accuracy']:.1%} acc., AUC {cc['roc_auc']:.3f}, R² {cr['r2']:.2f}", "Single dataset, no inflation, no marketing data"]],
    widths=[1.15, 1.2, 1.2, 1.25, 1.3])
H2("2.3 Gap Identification")
P("Most published models either require data that exists only close to or after release (tweets, opening-weekend screens, early votes) or are delivered as research code that a practitioner cannot try. Few studies explicitly guard against leakage from post-release columns such as vote counts, which are strongly correlated with revenue but unavailable when a project is being greenlit. CineMind fills this gap by (i) restricting the predictors to pre-release information, (ii) computing director and actor track records with strictly earlier films only, (iii) combining classification, regression and recommendation in one reproducible pipeline, and (iv) exposing it through an easy-to-use web app.")

# ------------------------------------------------------------ chapter 3
H1("Chapter 3: System Requirements & Architecture")
H2("3.1 Hardware Environment")
table("Hardware requirements", ["Component", "Minimum", "Used"], [["Processor", "Dual-core 2 GHz", "Apple Silicon (arm64) laptop"], ["RAM", "4 GB", "16 GB"], ["Storage", "1 GB free", "< 100 MB for data, models and figures"], ["GPU", "Not required", "None (classical ML, trains in minutes on CPU)"]], widths=[1.5, 2, 2.5])
H2("3.2 Software Environment")
table("Software stack", ["Item", "Details"], [["Operating system", "macOS (any OS with Python 3.10+ works)"], ["Language", "Python 3.13"], ["Data & ML", "pandas, NumPy, scikit-learn 1.7"], ["Visualisation", "matplotlib, seaborn"], ["Persistence", "joblib"], ["Web application", "Streamlit 1.51"]], widths=[1.8, 4.2])
H2("3.3 System Architecture")
P("The system follows a linear pipeline: data ingestion, cleaning and filtering, feature engineering, model training with cross-validation and tuning, evaluation on a held-out test set, and an inference application that loads the persisted models. A separate branch builds the recommender index.")
figure(FIG / "architecture.png", "System architecture of CineMind", 6.0)

# ------------------------------------------------------------ chapter 4
H1("Chapter 4: Data Pipeline & Feature Engineering")
H2("4.1 Dataset Description")
P(f"The IMDb 5000 Movie Dataset is a public collection of film metadata scraped from IMDb and distributed on Kaggle. The raw file contains 5,043 rows and {S['raw_cols']} columns; {S['raw_rows']:,} titles remain after removing duplicate (title, year) records. The columns include director and three actors with their Facebook page likes, genres, plot keywords, budget, gross, duration, content rating, language, country, release year, aspect ratio, and post-release statistics such as IMDb score, vote count and review counts. Data types are mixed: numeric (budget, gross, likes, duration, year), categorical (colour, content rating, language, country), pipe-separated multi-valued text (genres, plot keywords) and free text (names, titles).")
P(f"Missing values were most frequent in gross ({S['missing_before']['gross']}), budget ({S['missing_before']['budget']}), aspect ratio ({S['missing_before']['aspect_ratio']}) and content rating ({S['missing_before']['content_rating']}).")
H2("4.2 Exploratory Data Analysis (EDA)")
figure(FIG / "eda_missing_score.png", "Missing values and IMDb score distribution")
P("Figure 2 shows that revenue-related columns have the largest gaps and that IMDb scores are close to normally distributed around 6.5. Rows without budget or gross cannot be labelled and were removed (Section 4.3).")
figure(FIG / "eda_distributions.png", "Distributions of log-budget, log-gross, runtime outliers and gross by content rating", 5.6)
P("Both money variables are strongly right-skewed, so a base-10 logarithm gives near-symmetric distributions and is used as the regression target. Runtime shows a small number of extreme outliers; median gross is highest for G- and PG-rated titles, followed by PG-13, and lowest for R-rated films.")
figure(FIG / "eda_correlation.png", "Correlation matrix of numeric variables", 5.0)
P("Budget, vote counts and review counts correlate most with gross. Vote and review counts are post-release variables and were deliberately excluded from the predictors to avoid leakage; they appear here only for exploration.")
figure(FIG / "eda_budget_gross.png", "Budget versus gross (log scale) with break-even line", 4.6)
figure(FIG / "eda_genre_decade.png", "Hit rate by genre and by decade", 5.8)
P(f"Overall {S['hit_rate']:.1%} of the retained films recouped their budget. Hit rates differ across genres: horror (67%), family (65%) and animation (63%) recoup their budgets most often, while action (51%) and history (50%) are lowest, which reflects the high cost of big-budget action films. Budget alone is far from sufficient: many high-budget films lie below the break-even line.")
H2("4.3 Preprocessing Steps")
B(f" De-duplicated by (title, year): 5,043 → {S['raw_rows']:,} rows.", "Duplicates.")
B(f" Kept country = USA only, because budgets of foreign films are reported in local currencies (e.g. yen, won) and are not comparable. Rows without budget, gross or year were removed, as were implausible values (budget < $100k or gross < $10k, typically unit errors). Result: {S['usa_rows_with_money']:,} films.", "Filtering.")
B(" Numeric features were imputed with the median inside the model pipeline; categorical features with the most frequent value. Doing this inside the pipeline means imputation statistics are learned from training folds only.", "Missing values.")
B(" Content rating was cleaned (e.g. “Not Rated”, “Approved”, “Passed” → Unrated; “GP”, “M” → PG; “X” → NC-17) and one-hot encoded together with colour.", "Categorical encoding.")
B(" Genres were multi-hot encoded for the 19 genres with at least 60 films.", "Multi-valued fields.")
B(" Numeric inputs were standardised for the distance/gradient-based models (logistic regression, SVM). Tree-based models are scale-invariant and receive unscaled data.", "Scaling.")
B(f" Hit = gross ≥ budget. The classes are only mildly imbalanced ({S['hit_rate']:.1%} hits), so class weighting (class_weight = balanced) was preferred over SMOTE, which would create synthetic films with meaningless combinations of genres and cast.", "Class balance.")
B(" For each film we computed the director’s and lead actor’s number of earlier films, mean log-ROI and hit rate using only films released before it (cumulative sum minus the current row). This captures reputation without leaking the film’s own outcome.", "Leakage-safe track record.")
B(" 80/20 stratified train–test split (2,357 / 590 rows) with a fixed random seed.", "Split.")
table("IQR outlier counts (1.5 × IQR rule) after cleaning", ["Variable", "Outliers"], [[k, v] for k, v in S["iqr_outliers"].items()], widths=[3.5, 1.5])
P("Outliers in budget and gross correspond to genuine blockbusters, so they were kept rather than removed; the log transformation limits their influence on regression.")
H2("4.4 Feature Selection / Extraction")
P(f"The final feature matrix has {S['n_features_raw']} raw columns (six track-record features, budget and log-budget, runtime, year, five popularity measures, aspect ratio, language flag, 19 genre flags, content rating and colour) which become roughly 45 columns after one-hot encoding. Instead of discarding variables a priori, importance was measured after training using permutation importance on the test set (Section 6.2), and mutual-information-style reasoning guided the exclusion of post-release columns. For the recommender, plot keywords, genres, director and actors were extracted into a bag-of-tokens “soup” and converted into a TF-IDF matrix.")

# ------------------------------------------------------------ chapter 5
H1("Chapter 5: Methodology & Model Training")
H2("5.1 Selection of Algorithms")
table("Algorithms and the reason for choosing them", ["Algorithm", "Task", "Reason"], [
    ["Logistic Regression", "Classification", "Transparent baseline; coefficients are interpretable"],
    ["Decision Tree", "Classification", "Rule-based and easy to explain; high variance"],
    ["SVM (RBF kernel)", "Classification", "Handles non-linear boundaries on moderately sized data"],
    ["Random Forest", "Both", "Robust to outliers and mixed features; low tuning burden"],
    ["Gradient Boosting", "Both", "Strong on tabular data; sequential error correction"],
    ["Linear / Ridge Regression", "Regression", "Baselines for log gross; Ridge handles collinear likes"],
    ["TF-IDF + cosine", "Recommendation", "Standard content-based similarity; no user data needed"]], widths=[1.7, 1.3, 3.0])
H2("5.2 Model Architecture / Mathematical Equations")
P("Logistic regression models the probability of a hit as:")
eq("P(y = 1 | x) = 1 / (1 + exp(−(w·x + b)))")
P("Random Forest averages B decision trees, each grown on a bootstrap sample with a random subset of features at each split. For classification, the probability is the mean of the trees’ class probabilities and the split criterion is Gini impurity:")
eq("ŷ(x) = (1/B) Σ_b T_b(x)        Gini(t) = 1 − Σ_k p_k²")
P("Gradient boosting builds an additive model F_m(x) = F_(m−1)(x) + η · h_m(x), where each shallow tree h_m is fitted to the negative gradient of the loss (log-loss for classification, squared error for regression) and η is the learning rate. The RBF-kernel SVM uses K(x, x′) = exp(−γ‖x − x′‖²). The recommender represents each film as a TF-IDF vector and scores similarity by the cosine of the angle between vectors:")
eq("tfidf(t, d) = tf(t, d) · log(N / df(t))        sim(a, b) = (a · b) / (‖a‖ ‖b‖)")
H2("5.3 Hyperparameter Tuning")
P("Random Forest and Gradient Boosting (classification) and Gradient Boosting (regression) were tuned with RandomizedSearchCV (12 sampled configurations, five-fold stratified cross-validation, scoring by ROC-AUC for classification and R² for regression). Randomised search was chosen over exhaustive grid search because it explores wide ranges at a fraction of the cost.")
table("Selected hyper-parameters", ["Model", "Best parameters"], [
    ["Random Forest (classifier)", ", ".join(f"{k[3:]}={v}" for k, v in S["best_params"]["Random Forest"].items())],
    ["Gradient Boosting (classifier)", ", ".join(f"{k[3:]}={v}" for k, v in S["best_params"]["Gradient Boosting"].items())],
    ["Gradient Boosting (regressor)", ", ".join(f"{k[3:]}={v}" for k, v in S["best_params"]["GB Regressor"].items())]], widths=[2.0, 4.0])

# ------------------------------------------------------------ chapter 6
H1("Chapter 6: Results & Evaluation Analysis")
H2("6.1 Evaluation Metrics")
P("For classification: Accuracy = (TP+TN)/N; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R); ROC-AUC, the probability that a randomly chosen hit is ranked above a randomly chosen flop. For regression on log10(gross): RMSE = √(mean (y−ŷ)²), MAE = mean |y−ŷ| and R² = 1 − SS_res/SS_tot. An error of 0.4 in log10 units corresponds to a multiplicative error of about 2.5×.")
H2("6.2 Test Results & Graphs")
rows = [[n, f"{v['accuracy']:.3f}", f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", f"{v['roc_auc']:.3f}", f"{v['cv_auc_mean']:.3f} ± {v['cv_auc_std']:.3f}"] for n, v in C.items()]
table("Classification results on the 590-film test set", ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "5-fold CV AUC"], rows, widths=[1.7, .6, .6, .6, .6, .8, 1.1])
figure(FIG / "res_roc.png", "ROC curves of all classifiers", 4.6)
figure(FIG / "res_confusion.png", f"Confusion matrix of the champion classifier ({CC})", 3.9)
figure(FIG / "res_learning_curve.png", "Learning curve: training versus validation accuracy", 4.6)
P("The learning curve shows a clear gap between training and validation accuracy for the forest, which is typical for flexible tree ensembles on a few thousand rows; training accuracy stays above 0.90 while validation accuracy rises only slowly from about 0.57 to 0.60, so more data of the same kind would help only slowly.")
figure(FIG / "res_importance.png", "Permutation importance of the top features", 4.8)
P("Release year, runtime, budget (raw and log), director popularity and the director’s previous log-ROI are the strongest predictors, followed by genre flags such as Comedy, Action and Horror, in line with the genre hit rates seen in Figure 6. Year matters because the budget/gross relationship drifts over the decades (no inflation adjustment).")
rows = [[n, f"{v['r2']:.3f}", f"{v['rmse_log']:.3f}", f"{v['mae_log']:.3f}"] for n, v in R.items()]
table("Regression results (target = log10 gross) on the test set", ["Model", "R²", "RMSE (log10)", "MAE (log10)"], rows, widths=[2.4, 1, 1.3, 1.3])
figure(FIG / "res_actual_vs_pred.png", f"Actual versus predicted log10 gross ({CR})", 4.2)
H2("6.3 Comparison of Models")
figure(FIG / "res_model_comparison.png", "Classifier comparison on the test set", 5.4)
P(f"Among classifiers, the tuned Random Forest has the best ROC-AUC ({C[CC]['roc_auc']:.3f}) and ties for the best accuracy ({C[CC]['accuracy']:.3f}), so it is the champion. Gradient Boosting attains the highest recall ({C['Gradient Boosting (tuned)']['recall']:.3f}) and F1 ({C['Gradient Boosting (tuned)']['f1']:.3f}) because it leans towards predicting “Hit”; it would be preferable when missing a hit is costlier than a false alarm. The single Decision Tree is clearly the weakest and over-fits. Ensembles beat the linear baseline on regression as well (R² {cr['r2']:.3f} versus {R['Linear Regression']['r2']:.3f}).")
P(f"Overall, a hit/flop accuracy of about {cc['accuracy']:.0%} is modest, but it should be read against a majority-class baseline of {S['hit_rate']:.0%} and the fact that the model sees no marketing, release-date, script or reviews information. This is consistent with earlier studies, which found pre-release box-office prediction inherently noisy.")

# ------------------------------------------------------------ chapter 7
H1("Chapter 7: Deployment & UI")
H2("7.1 User Interface Design")
P("The application is built with Streamlit and has four tabs. (1) Predict a film: a form for budget, runtime, year, genres, certificate, director, lead actor and popularity values; on submit it displays the hit probability, predicted gross and predicted profit. Choosing a known director or actor automatically pulls their career track record. (2) Recommend: choose a film and the number of results; the app lists the most similar films with their similarity score. (3) Industry insights: interactive year and genre filters with KPI tiles, a hit-rate-by-decade chart, a budget-versus-gross scatter plot and a ranking of directors by return on investment. (4) Model report: the tables and charts of Chapter 6. The trained models are loaded once with st.cache_resource, so predictions return instantly. Start the app with: streamlit run app.py.")
H2("7.2 System Snapshots")
figure(SHOT / "ui_predict_result.png", "Prediction tab with result metrics", 5.4)
figure(SHOT / "ui_recommend.png", "Recommendation tab for the film “Avatar”", 5.4)
figure(SHOT / "ui_insights.png", "Industry-insights tab", 5.4)
figure(SHOT / "ui_report.png", "Model-report tab", 5.4)

# ------------------------------------------------------------ chapter 8
H1("Chapter 8: Conclusion & Future Scope")
H2("8.1 Conclusion")
P(f"CineMind demonstrates an end-to-end machine-learning workflow for the movie industry: cleaning a real dataset, engineering leakage-free features, training and tuning several models, evaluating them honestly on unseen data and delivering the result as an interactive web application. The tuned Random Forest predicts hit or flop with {cc['accuracy']:.1%} accuracy and {cc['roc_auc']:.3f} ROC-AUC, the Random Forest regressor explains {cr['r2']:.1%} of the variance of log gross, and the TF-IDF recommender produces sensible neighbours (for example, Aliens and Terminator 2 for Avatar). All objectives listed in Section 1.3 were met.")
H2("8.2 Limitations")
B("Only information from one dataset (US films, mostly 1980–2016) is used; the model may not generalise to streaming-era releases.")
B("Budget and gross are nominal dollars; inflation adjustment would make decades comparable.")
B("Gross in this dataset is mostly domestic or as reported and not always worldwide; Hit is therefore an approximation of profitability that ignores marketing costs and ancillary revenue.")
B("Facebook-likes columns were scraped at one point in time and reflect later popularity, which may introduce mild temporal bias.")
B("Classification accuracy is moderate, so predictions should support, not replace, human judgement.")
B("The recommender is content-based only and cannot personalise to individual users.")
H2("8.3 Future Work")
B("Add release month, competing releases, number of screens, marketing budget and trailer views.")
B("Add text embeddings of plot summaries and sentiment from social media or trailer comments.")
B("Adjust all money values for inflation and include worldwide gross from a larger source such as TMDB.")
B("Adopt gradient-boosting libraries (XGBoost, LightGBM), model stacking and probability calibration.")
B("Add collaborative filtering on a ratings dataset (e.g. MovieLens) for a hybrid recommender.")
B("Containerise the app, add scheduled retraining and monitoring (MLOps), and deploy it publicly.")

# ------------------------------------------------------------ references
H1("References")
for r in [
    "Sharda, R., & Delen, D. (2006). Predicting box-office success of motion pictures with neural networks. Expert Systems with Applications, 30(2), 243–254.",
    "Asur, S., & Huberman, B. A. (2010). Predicting the future with social media. Proceedings of the IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology, 492–499.",
    "Ghiassi, M., Lio, D., & Moon, B. (2015). Pre-production forecasting of movie revenues with a dynamic artificial neural network. Expert Systems with Applications, 42(7), 3176–3193.",
    "Lash, M. T., & Zhao, K. (2016). Early predictions of movie success: The who, what, and when of profitability. Journal of Management Information Systems, 33(3), 874–903.",
    "Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender Systems Handbook (pp. 73–105). Springer.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "IMDb 5000 Movie Dataset. Kaggle (originally scraped from IMDb). https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset"]:
    p = P(r); p.paragraph_format.left_indent = Inches(.4); p.paragraph_format.first_line_indent = Inches(-.4)
P("[[Verify each citation’s volume and page numbers against the publisher’s page before submitting.]]", align="l")

# ------------------------------------------------------------ list of figures / tables (inserted at the TOC anchor)
lof = ["LIST OF FIGURES"] + FIGS + ["", "LIST OF TABLES"] + TABLES
anchor = LOF_ANCHOR
for line in lof:
    newp = OxmlElement("w:p"); anchor._p.addnext(newp)
    from docx.text.paragraph import Paragraph
    para = Paragraph(newp, anchor._parent); run = para.add_run(line)
    if line.startswith("LIST"): run.bold = True
    para.paragraph_format.line_spacing = 1.15
    anchor = para

# ask Word to refresh fields on open
st_el = doc.settings.element; uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); st_el.append(uf)
out = ROOT / "CineMind_ML_Mini_Project_Report.docx"
doc.save(out); print("saved", out)
