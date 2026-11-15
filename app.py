import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, silhouette_score)

st.set_page_config(page_title="EduGuard", page_icon="🎓", layout="wide")


# =====================================================================
# DATA
# =====================================================================
st.title("EduGuard 🎓")
st.caption("An ML arena that finds at-risk students early, and shows what can be done about it.")

file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
if file is not None:
    df = pd.read_csv(file)
else:
    st.sidebar.info("No file uploaded, so using the sample data.")
    df = pd.read_csv("students.csv")

# ---- Data Doctor ----
missing = int(df.isna().sum().sum())
df = df.dropna().reset_index(drop=True)
doctor_notes = []
if missing > 0:
    doctor_notes.append(f"Found {missing} missing values. Rows with gaps were removed.")
if len(df) < 60:
    doctor_notes.append(f"Only {len(df)} rows. Results will be shaky. Try to collect 100 or more.")

target = st.sidebar.selectbox("Column to predict", df.columns, index=len(df.columns) - 1)
classes = list(df[target].unique())
if len(classes) != 2:
    st.error("This version needs a target with exactly 2 classes (e.g. Safe / At Risk).")
    st.stop()
default_pos = classes.index("At Risk") if "At Risk" in classes else 0
pos = st.sidebar.selectbox("Which class is the risky one?", classes, index=default_pos)

if df[target].value_counts().min() < 10:
    doctor_notes.append("One class has fewer than 10 students, so cross-validation may fail or mislead.")

X = df.drop(columns=[target]).select_dtypes(include="number")
y = (df[target] == pos).astype(int)

# =====================================================================
# SIDEBAR: VIEW + MODEL SETTINGS
# =====================================================================
st.sidebar.divider()
view = st.sidebar.radio("Who is using this?", ["👩‍🏫 Teacher", "🎓 Student"])

st.sidebar.header("Model settings")
k = st.sidebar.slider("kNN: k", 1, 25, 5)
depth = st.sidebar.slider("Tree: max depth", 1, 15, 3)
kernel = st.sidebar.selectbox("SVM: kernel", ["rbf", "linear", "poly"])
C = st.sidebar.select_slider("SVM: C", options=[0.01, 0.1, 1, 10, 100], value=1)
gamma = st.sidebar.select_slider("SVM: gamma", options=[0.001, 0.01, 0.1, 1, 10], value=0.1)
act = st.sidebar.selectbox("Neural Net: activation", ["relu", "tanh", "logistic"])
h1 = st.sidebar.slider("Neural Net: neurons in layer 1", 4, 64, 16)
h2 = st.sidebar.slider("Neural Net: neurons in layer 2", 4, 64, 8)


# =====================================================================
# TRAINING (cached, so moving a tab does not retrain everything)
# =====================================================================
@st.cache_resource(show_spinner="Training all models...")
def train_all(_X, _y, data_key, k, depth, kernel, C, gamma, act, h1, h2):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y)           # Exp 1
    models = {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
        "Decision Tree": DecisionTreeClassifier(max_depth=depth, random_state=42,
                                                class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight="balanced")),
        "SVM": make_pipeline(StandardScaler(), SVC(
            kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42,
            class_weight="balanced")),
        "Neural Network": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(h1, h2), activation=act, max_iter=1000, random_state=42)),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)    # k-fold (Exp 1)
    rows, probas, gaps = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        probas[name] = prob
        tr = accuracy_score(y_train, model.predict(X_train))
        te = accuracy_score(y_test, pred)
        gaps[name] = tr - te
        rows.append({
            "Model": name,
            "Accuracy": te,
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": roc_auc_score(y_test, prob),
            "CV F1 (5-fold)": cross_val_score(model, _X, _y, cv=skf, scoring="f1").mean(),
        })
    score = pd.DataFrame(rows).set_index("Model")
    return dict(models=models, score=score, probas=probas, gaps=gaps,
                X_test=X_test, y_test=y_test, winner=score["CV F1 (5-fold)"].idxmax())


@st.cache_data(show_spinner=False)
def cluster_curves(_X, data_key):
    Xs = StandardScaler().fit_transform(_X)
    ks = list(range(2, 9))
    inertias, sils = [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
    return ks, inertias, sils


data_key = str(pd.util.hash_pandas_object(df, index=True).sum()) + target + str(pos)
R = train_all(X, y, data_key, k, depth, kernel, C, gamma, act, h1, h2)
models, score, winner = R["models"], R["score"], R["winner"]
model = models[winner]
risk_all = model.predict_proba(X)[:, 1]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def risk_of(row):
    return float(model.predict_proba(pd.DataFrame([row])[X.columns])[0][1])


def reasons_for(row, top=3):
    base = risk_of(row)
    out = []
    for f in X.columns:
        r = row.copy()
        r[f] = X[f].mean()
        out.append((f, float(row[f]), float(X[f].mean()), base - risk_of(r)))
    out.sort(key=lambda t: abs(t[3]), reverse=True)
    return base, out[:top]


def find_fix(row, actionable):
    """Greedy search for the smallest changes that bring risk below 50%."""
    start = row.astype(float)
    cur = start.copy()
    step = (X.max() - X.min()) * 0.03
    p = risk_of(cur)
    start_p = p
    for _ in range(150):
        if p < 0.5:
            break
        cands = []
        for f in actionable:
            for d in (1, -1):
                c = cur.copy()
                c[f] = np.clip(c[f] + d * step[f], X[f].min(), X[f].max())
                if c[f] != cur[f]:
                    cands.append(c)
        if not cands:
            break
        probs = model.predict_proba(pd.DataFrame(cands)[X.columns])[:, 1]
        j = int(probs.argmin())
        if probs[j] >= p - 1e-9:
            break
        cur, p = cands[j], float(probs[j])
    return start, cur, start_p, p


def show_risk(p, label):
    if p >= 0.5:
        st.error(f"⚠️ {label}: risk {p*100:.0f}%")
    else:
        st.success(f"✅ Not {label}: risk {p*100:.0f}%")


# =====================================================================
# TEACHER VIEW
# =====================================================================
if view.startswith("👩"):
    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🏁 Model Arena", "👥 Groups",
                                  "⚖️ Fairness & Ethics", "📘 Learn"])

    # ---------- Overview ----------
    with t1:
        for note in doctor_notes:
            st.warning("Data Doctor: " + note)
        a, b, c = st.columns(3)
        a.metric("Students", len(df))
        b.metric("Columns", df.shape[1])
        c.metric("At-risk share", f"{y.mean()*100:.0f}%")

        st.subheader("Data preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Class balance")
        st.bar_chart(df[target].value_counts())

        st.subheader("🚩 Students the model flags as at risk")
        flagged = X.assign(risk_percent=(risk_all * 100).round(0))
        flagged = flagged[risk_all >= 0.5].sort_values("risk_percent", ascending=False)
        st.write(f"{len(flagged)} of {len(df)} students flagged by **{winner}**. "
                 "A flag means *check in and offer help*, not *punish*.")
        st.dataframe(flagged, use_container_width=True)

    # ---------- Model Arena ----------
    with t2:
        st.subheader("🏆 Scoreboard")
        st.dataframe(score.style.format("{:.3f}").highlight_max(axis=0, color="#14532d"),
                     use_container_width=True)
        st.success(f"Winner: **{winner}** (best 5-fold cross-validated F1 = "
                   f"{score.loc[winner, 'CV F1 (5-fold)']:.3f})")
        st.caption("F1 is used instead of plain accuracy because At Risk students are fewer than Safe ones. "
                   "The test set is small, so trust the CV F1 column most.")

        st.subheader("🚨 Overfitting alarm (train vs test accuracy)")
        for name, g in R["gaps"].items():
            if g > 0.10:
                st.error(f"{name}: train is {g*100:.0f} points higher than test, so it is memorizing.")
            else:
                st.write(f"✅ {name}: gap is only {g*100:.1f} points.")

        left, right = st.columns(2)
        with left:
            st.subheader("ROC curves")
            fig, ax = plt.subplots(figsize=(6, 5))
            for name, prob in R["probas"].items():
                fpr, tpr, _ = roc_curve(R["y_test"], prob)
                ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(R['y_test'], prob):.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=8)
            st.pyplot(fig)
        with right:
            st.subheader("What matters most")
            pi = permutation_importance(model, R["X_test"], R["y_test"], scoring="f1",
                                        n_repeats=20, random_state=42)
            imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(imp.index, imp.values)
            ax.set_xlabel(f"Drop in F1 when shuffled ({winner})")
            st.pyplot(fig)

        with st.expander("See the decision tree (the rules behind the risk)"):
            if depth <= 4:
                fig, ax = plt.subplots(figsize=(14, 6))
                plot_tree(models["Decision Tree"], feature_names=list(X.columns),
                          class_names=["Not " + str(pos), str(pos)], filled=True,
                          fontsize=8, ax=ax)
                st.pyplot(fig)
            else:
                st.caption("Set tree depth to 4 or less in the sidebar to see the picture.")

    # ---------- Groups ----------
    with t3:
        st.subheader("Student groups (K-Means)")
        ks, inertias, sils = cluster_curves(X, data_key)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
        a1.plot(ks, inertias, "o-"); a1.set_title("Elbow method"); a1.set_xlabel("k")
        a2.plot(ks, sils, "o-"); a2.set_title("Silhouette score"); a2.set_xlabel("k")
        st.pyplot(fig)
        n_clusters = st.slider("Number of groups", 2, 8, 3)
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Xs)
        g = X.copy()
        g["group"] = km.labels_
        g["at_risk"] = y.values
        summary = g.groupby("group").mean().round(2)
        summary["students"] = g.groupby("group").size()
        st.write("Average values per group. The group with the highest `at_risk` is the struggling group.")
        st.dataframe(summary, use_container_width=True)

    # ---------- Fairness & Ethics ----------
    with t4:
        st.subheader("Bias checker")
        st.write("Does the model work equally well for different groups of students?")
        split_col = st.selectbox("Split students by", list(X.columns), key="bias_col")
        cut = st.slider("Group boundary", float(X[split_col].min()), float(X[split_col].max()),
                        float(X[split_col].median()), key="bias_cut")
        pred_all = model.predict(X)
        rows = []
        for label, mask in [(f"{split_col} below {cut:.1f}", X[split_col] < cut),
                            (f"{split_col} at or above {cut:.1f}", X[split_col] >= cut)]:
            if mask.sum() == 0:
                continue
            yt, yp = y[mask], pred_all[mask.values]
            rows.append({"Group": label, "Students": int(mask.sum()),
                         "Actually at risk": int(yt.sum()),
                         "Accuracy": accuracy_score(yt, yp),
                         "Recall": recall_score(yt, yp, zero_division=0),
                         "Flagged as at risk": f"{yp.mean()*100:.0f}%"})
        bias_df = pd.DataFrame(rows).set_index("Group")
        st.dataframe(bias_df.style.format({"Accuracy": "{:.2f}", "Recall": "{:.2f}"}),
                     use_container_width=True)
        if len(bias_df) == 2:
            diff = abs(bias_df["Recall"].iloc[0] - bias_df["Recall"].iloc[1])
            if diff > 0.15:
                st.error(f"⚠️ Recall differs by {diff*100:.0f} points between the groups, "
                         "so the model may be unfair to one group.")
            else:
                st.success("✅ Recall is similar across both groups.")
        st.caption("If a group has very few at-risk students, the numbers are unreliable.")

        st.subheader("Ethics and limitations")
        st.markdown("""
- **This tool supports people, it does not judge them.** A flag means "check in and offer help", never "punish".
- **Predictions come from patterns, not certainty.** Some flagged students will be fine, and some missed students will struggle.
- **Small data is shaky.** With only a few hundred rows, results can change a lot between runs.
- **Privacy:** data is anonymous, contains no names or roll numbers, and is used only for this project.
- **Fairness:** always check the Bias checker before trusting the results for any group.
- **Advice is not a guarantee.** The "How to fix it" suggestions show what the model learned, not what will certainly happen.
""")

    # ---------- Learn ----------
    with t5:
        st.subheader("The algorithms in this project")
        st.markdown("""
| Model | Lab experiment | In one line |
|---|---|---|
| Train/test split, k-fold CV | Exp 1 | Test on students the model has never seen, and repeat 5 times for fairness |
| kNN | Exp 4 | Look at the k most similar students and take a vote |
| Decision Tree | Exp 3 | Asks yes/no questions; too deep means it memorizes (overfitting) |
| Naive Bayes | Exp 7 | Uses probabilities, assuming features are independent |
| Logistic Regression | Exp 8 | Draws a straight boundary and gives a risk probability (sigmoid) |
| SVM | Exp 9 | Finds the widest-margin boundary; C and gamma control how tight it is |
| Neural Network | Exp 10 | Layers of neurons that learn curved patterns |
| K-Means | Exp 14 | Groups similar students without being told the answer |
| ROC and AUC | Exp 8 | How well a model separates at-risk from safe at every threshold |
""")

# =====================================================================
# STUDENT VIEW
# =====================================================================
else:
    st.subheader("Enter your details")
    st.caption("Nothing is saved. Values start at the class average.")
    me = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        me[col] = cols[i % 3].number_input(
            col, min_value=float(X[col].min()), max_value=float(X[col].max()),
            value=float(round(X[col].mean(), 1)), step=1.0, key=f"me_{col}")
    me = pd.Series(me)[X.columns]

    s1, s2, s3 = st.tabs(["🎯 My risk", "🎚️ What-if", "🛠️ How to fix it"])

    # ---------- My risk ----------
    with s1:
        base, top = reasons_for(me)
        st.metric("Your risk", f"{base*100:.0f}%")
        show_risk(base, str(pos))
        st.write(f"Prediction by **{winner}**. This is a pattern-based estimate, not a verdict.")
        st.subheader("Why?")
        shown = False
        for f, val, avg, effect in top:
            if effect > 0.01:
                st.write(f"🔺 **{f}** = {val:.1f} (class average {avg:.1f}) pushes your risk **up** by about {effect*100:.0f} points")
                shown = True
            elif effect < -0.01:
                st.write(f"🔻 **{f}** = {val:.1f} (class average {avg:.1f}) pushes your risk **down** by about {-effect*100:.0f} points")
                shown = True
        if not shown:
            st.write("No single factor stands out.")

    # ---------- What-if ----------
    with s2:
        st.write("Move the sliders to see how your risk would change.")
        new = {}
        sc = st.columns(3)
        for i, col in enumerate(X.columns):
            lo, hi = float(X[col].min()), float(X[col].max())
            new[col] = sc[i % 3].slider(col, lo, hi, float(me[col]), key=f"wi_{col}")
        new = pd.Series(new)[X.columns]
        before, after = risk_of(me), risk_of(new)
        m1, m2 = st.columns(2)
        m1.metric("Original risk", f"{before*100:.0f}%")
        m2.metric("New risk", f"{after*100:.0f}%", delta=f"{(after-before)*100:.0f}%",
                  delta_color="inverse")
        if after >= 0.5:
            st.error("Still in the at-risk zone. Try raising attendance, study hours or marks.")
        else:
            st.success("Out of the at-risk zone with these changes!")

    # ---------- How to fix it ----------
    with s3:
        st.write("The app searches for the smallest changes that bring your risk below 50%.")
        default_actionable = [c for c in X.columns
                              if c in ("attendance", "study_hours", "assignments_pct", "sleep_hours")]
        actionable = st.multiselect("Things you can change", list(X.columns),
                                    default=default_actionable or list(X.columns))
        if risk_of(me) < 0.5:
            st.success("You are not in the at-risk zone, so there is nothing to fix. Keep it up!")
        elif not actionable:
            st.info("Pick at least one thing you can change.")
        else:
            start, cur, p0, p1 = find_fix(me, actionable)
            ch = pd.DataFrame({"Now": start[actionable].round(1),
                               "Suggested": cur[actionable].round(1)})
            ch["Change"] = (ch["Suggested"] - ch["Now"]).round(1)
            ch = ch[ch["Change"] != 0]
            a, b = st.columns(2)
            a.metric("Risk now", f"{p0*100:.0f}%")
            b.metric("Risk after changes", f"{p1*100:.0f}%", delta=f"{(p1-p0)*100:.0f}%",
                     delta_color="inverse")
            if p1 < 0.5:
                st.success("These changes move you out of the at-risk zone:")
                for f, row in ch.iterrows():
                    word = "increase" if row["Change"] > 0 else "reduce"
                    st.write(f"• {word} **{f}** from {row['Now']} to {row['Suggested']}")
            else:
                st.warning("These factors alone are not enough. Try adding more to the list above, "
                           "and talk to your mentor.")
                if len(ch):
                    st.dataframe(ch)
            st.caption("Advice comes from patterns in the data, not a guaranteed outcome.")