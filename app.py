import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, silhouette_score
from engine import clean, train_all, make_pre, focus_column

st.set_page_config(page_title="ML Model Arena", page_icon="🏟️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {display: none;}
</style>""", unsafe_allow_html=True)
st.title("ML Model Arena 🏟️")
st.caption("Upload any CSV, pick the column to predict, and let six models compete. "
           "Then explore, explain and test what-if changes.")


# =====================================================================
# 1. DATA (any CSV, or a built-in demo)
# =====================================================================
def make_space_data():
    """Simulated astronaut bone-loss data. Built from the seminar figures: about 1-1.5% bone
    loss per month in space, reduced by exercise (about 2 hours per day is the usual plan).
    No public astronaut dataset exists, so this is a teaching simulation, not real crew data."""
    rng = np.random.default_rng(7)
    n = 400
    months = rng.integers(1, 31, n)
    exercise = rng.normal(1.6, 0.6, n).clip(0.2, 3.0)
    resistive = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    age = rng.integers(30, 56, n)
    sex = rng.choice(["Female", "Male"], n)
    calcium = rng.normal(1000, 200, n).clip(500, 1600)
    vit_d = rng.normal(800, 250, n).clip(200, 1500)
    sleep = rng.normal(6.2, 0.9, n).clip(4, 9)
    rate = (1.4 - 0.35 * exercise - 0.25 * (resistive == "Yes") - 0.0003 * (calcium - 1000)
            - 0.0002 * (vit_d - 800) + 0.004 * (age - 40) + rng.normal(0, 0.12, n)).clip(0.1, 1.6)
    loss = rate * months
    risk = np.where(loss < 5, "Low", np.where(loss < 12, "Moderate", "High"))
    return pd.DataFrame({
        "mission_months": months, "exercise_hours_per_day": exercise.round(2),
        "resistive_exercise": resistive, "age": age, "sex": sex,
        "calcium_mg": calcium.round(0), "vitamin_d_iu": vit_d.round(0),
        "sleep_hours": sleep.round(1), "bone_loss_risk": risk})


@st.cache_data(show_spinner=False)
def load_demo(name):
    from sklearn import datasets
    if name.startswith("Space"):
        return make_space_data()
    if name.startswith("Iris"):
        d = datasets.load_iris(as_frame=True)
        df = d.frame.drop(columns="target")
        df["species"] = d.target.map(dict(enumerate(d.target_names)))
        return df
    d = datasets.load_breast_cancer(as_frame=True)
    df = d.frame.drop(columns="target")
    df["diagnosis"] = d.target.map({0: "malignant", 1: "benign"})
    return df


@st.cache_data(show_spinner=False)
def clean_cached(df, target):
    return clean(df, target)


file = st.sidebar.file_uploader("Upload your own CSV", type="csv")
demo = st.sidebar.selectbox("...or try a demo dataset",
                            ["Breast cancer (30 features)", "Iris flowers (3 classes)",
                             "Space health: bone-loss risk (simulated)"])
if file is not None:
    try:
        raw = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read this file as a CSV: {e}")
        st.stop()
    source = file.name
else:
    raw = load_demo(demo)
    source = demo
st.sidebar.caption(f"Using: **{source}**")

target = st.sidebar.selectbox("Column to predict", list(raw.columns), index=len(raw.columns) - 1)
if raw[target].nunique(dropna=True) < 2:
    st.error("The chosen column has fewer than 2 different values, so there is nothing to predict.")
    st.stop()
if raw[target].nunique(dropna=True) > 20:
    st.error("This column has more than 20 different values, so it looks like a number to estimate "
             "(regression) rather than a category. Pick a column with a small set of labels.")
    st.stop()

X, y, classes, num_cols, cat_cols, notes = clean_cached(raw, target)
if len(X.columns) == 0:
    st.error("No usable feature columns were left after cleaning.")
    st.stop()

counts = y.value_counts()
default_pos = classes.index(counts.idxmin())
pos = st.sidebar.selectbox("Class to focus on", classes, index=default_pos,
                           help="The outcome you care about most (for example 'At Risk' or 'churn'). "
                                "Probabilities, what-if and flagged lists use this class.")
multi = len(classes) > 2
if counts.min() < 10:
    notes.append(f"One class has only {counts.min()} rows, so results may be unreliable.")
if len(X) < 60:
    notes.append(f"Only {len(X)} rows. Results will be shaky.")

# =====================================================================
# 2. SIDEBAR: MODE + MODEL SETTINGS
# =====================================================================
st.sidebar.divider()
mode = st.sidebar.radio("Mode", ["📊 Analyst (compare models)", "🔮 Predictor (predict and explain)"])
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
# 3. TRAIN (cached)
# =====================================================================
@st.cache_resource(show_spinner="Training all models...")
def get_results(_X, _y, key, classes_t, pos, num_t, cat_t, k, depth, kernel, C, gamma, act, h1, h2):
    p = dict(k=k, depth=depth, kernel=kernel, C=C, gamma=gamma, act=act, h1=h1, h2=h2)
    return train_all(_X, _y, list(classes_t), pos, list(num_t), list(cat_t), p)


@st.cache_data(show_spinner=False)
def cluster_curves(_X, key, num_t, cat_t):
    Z = make_pre(list(num_t), list(cat_t), True).fit_transform(_X)
    ks = list(range(2, min(9, len(_X) - 1)))
    inertias, sils = [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_))
    return ks, inertias, sils


key = str(pd.util.hash_pandas_object(X, index=True).sum()) + target + str(pos) + str(len(classes))
try:
    R = get_results(X, y, key, tuple(classes), pos, tuple(num_cols), tuple(cat_cols),
                    k, depth, kernel, C, gamma, act, h1, h2)
except ValueError as e:
    st.error(f"Could not train the models: {e}")
    st.stop()

models, score, winner, yy = R["models"], R["score"], R["winner"], R["yy"]
model = models[winner]
fc = focus_column(classes, pos)
F1_NAME = "macro F1" if multi else "F1"

typical = {c: (X[c].median() if c in num_cols else X[c].mode().iloc[0]) for c in X.columns}
pred_idx_all = model.predict(X)
prob_all = model.predict_proba(X)[:, fc]


# =====================================================================
# 4. HELPERS
# =====================================================================
def rows_df(rows):
    return pd.DataFrame(rows)[list(X.columns)]


def prob_of(row):
    return float(model.predict_proba(rows_df([row]))[0][fc])


def reasons_for(row, top=3):
    base = prob_of(row)
    out = []
    for c in X.columns:
        r = dict(row)
        r[c] = typical[c]
        out.append((c, row[c], typical[c], base - prob_of(r)))
    out.sort(key=lambda t: abs(t[3]), reverse=True)
    return base, out[:top]


def desired_idx(label):
    return (1 if label == pos else 0) if not multi else classes.index(label)


def find_change(row, actionable, want):
    """Greedy search for the smallest changes that make the model predict `want`."""
    di = desired_idx(want)
    cur = dict(row)
    span = {c: (X[c].max() - X[c].min()) * 0.03 for c in num_cols}
    p = float(model.predict_proba(rows_df([cur]))[0][di])
    start_p = p
    for _ in range(150):
        if int(model.predict(rows_df([cur]))[0]) == di:
            break
        cands = []
        for c in actionable:
            if c in num_cols:
                for d in (1, -1):
                    new = dict(cur)
                    new[c] = float(np.clip(cur[c] + d * span[c], X[c].min(), X[c].max()))
                    if new[c] != cur[c]:
                        cands.append(new)
            else:
                for v in sorted(X[c].dropna().unique().tolist()):
                    if v != cur[c]:
                        new = dict(cur)
                        new[c] = v
                        cands.append(new)
        if not cands:
            break
        probs = model.predict_proba(rows_df(cands))[:, di]
        j = int(probs.argmax())
        if probs[j] <= p + 1e-9:
            break
        cur, p = cands[j], float(probs[j])
    ok = int(model.predict(rows_df([cur]))[0]) == di
    return cur, start_p, p, ok


def input_widgets(prefix, start=None, shown=None, slider=False):
    """One widget per feature. Features not in `shown` keep their start value."""
    vals = {}
    cols = st.columns(3)
    i = 0
    for c in X.columns:
        base = typical[c] if start is None else start[c]
        if shown is not None and c not in shown:
            vals[c] = base
            continue
        box = cols[i % 3]
        i += 1
        if c in num_cols:
            lo, hi = float(X[c].min()), float(X[c].max())
            base = float(min(max(base, lo), hi))
            if slider:
                vals[c] = box.slider(c, lo, hi, base, key=f"{prefix}_{c}")
            else:
                vals[c] = box.number_input(c, min_value=lo, max_value=hi, value=base,
                                           step=(hi - lo) / 100 if hi > lo else 1.0,
                                           key=f"{prefix}_{c}")
        else:
            opts = sorted(X[c].dropna().unique().tolist())
            vals[c] = box.selectbox(c, opts, index=opts.index(base) if base in opts else 0,
                                    key=f"{prefix}_{c}")
    return vals


# =====================================================================
# 5. ANALYST MODE
# =====================================================================
if mode.startswith("📊"):
    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🏁 Model Arena", "👥 Groups",
                                  "⚖️ Fairness & Ethics", "📘 Learn"])

    with t1:
        for n in notes:
            st.warning("Data Doctor: " + n)
        a, b, c = st.columns(3)
        a.metric("Rows used", len(X))
        b.metric("Feature columns", len(X.columns))
        c.metric("Classes", len(classes))
        st.subheader("Data preview")
        st.dataframe(raw.head(10))
        st.subheader("Class balance")
        st.bar_chart(y.astype(str).value_counts())
        st.subheader(f"Rows the model predicts as '{pos}'")
        flagged = X.assign(**{f"probability of {pos}": (prob_all * 100).round(0)})
        flagged = flagged[pred_idx_all == fc].sort_values(f"probability of {pos}", ascending=False)
        st.write(f"{len(flagged)} of {len(X)} rows are predicted as **{pos}** by **{winner}**.")
        st.dataframe(flagged.head(200))

    with t2:
        st.subheader("🏆 Scoreboard")
        st.dataframe(score.style.format("{:.3f}").highlight_max(axis=0, color="#14532d"))
        st.success(f"Winner: **{winner}** (best {R['folds']}-fold cross-validated {F1_NAME} = "
                   f"{score.loc[winner, 'CV F1']:.3f})")
        st.caption(f"{F1_NAME} is used instead of plain accuracy because classes can be unbalanced. "
                   "The test split is small, so trust the CV column most.")

        st.subheader("🚨 Overfitting alarm (train vs test accuracy)")
        for name, g in R["gaps"].items():
            if g > 0.10:
                st.error(f"{name}: train is {g*100:.0f} points higher than test, so it is memorizing.")
            else:
                st.write(f"✅ {name}: gap is only {g*100:.1f} points.")

        left, right = st.columns(2)
        with left:
            st.subheader(f"ROC curves for '{pos}'")
            fig, ax = plt.subplots(figsize=(6, 5))
            yt = (R["y_test"] == fc).astype(int)
            for name, prob in R["probas"].items():
                if yt.nunique() < 2:
                    break
                fpr, tpr, _ = roc_curve(yt, prob[:, fc])
                ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yt, prob[:, fc]):.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=8)
            st.pyplot(fig)
        with right:
            st.subheader("What matters most")
            pi = permutation_importance(model, R["X_test"], R["y_test"], scoring=R["scoring"],
                                        n_repeats=10, random_state=42)
            imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(12)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(imp.index, imp.values)
            ax.set_xlabel(f"Drop in {F1_NAME} when shuffled ({winner})")
            st.pyplot(fig)

        with st.expander("See the decision tree (the rules behind the prediction)"):
            if depth <= 4:
                tree = models["Decision Tree"]
                names = [n.split("__", 1)[-1] for n in tree[:-1].get_feature_names_out()]
                cn = [str(c) for c in classes] if multi else ["Not " + str(pos), str(pos)]
                fig, ax = plt.subplots(figsize=(14, 6))
                plot_tree(tree[-1], feature_names=names, class_names=cn,
                          filled=True, fontsize=8, ax=ax)
                st.pyplot(fig)
            else:
                st.caption("Set tree depth to 4 or less in the sidebar to see the picture.")

    with t3:
        st.subheader("Groups found without labels (K-Means)")
        if len(X) < 10:
            st.info("Not enough rows for clustering.")
        else:
            ks, inertias, sils = cluster_curves(X, key, tuple(num_cols), tuple(cat_cols))
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
            a1.plot(ks, inertias, "o-"); a1.set_title("Elbow method"); a1.set_xlabel("k")
            a2.plot(ks, sils, "o-"); a2.set_title("Silhouette score"); a2.set_xlabel("k")
            st.pyplot(fig)
            n_clusters = st.slider("Number of groups", 2, 8, 3)
            Z = make_pre(num_cols, cat_cols, True).fit_transform(X)
            labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(Z)
            summ = pd.DataFrame({"rows": pd.Series(labels).value_counts().sort_index()})
            summ[f"share that is '{pos}' (%)"] = ((yy == fc).groupby(labels).mean() * 100).round(0)
            if num_cols:
                summ = summ.join(X[num_cols].groupby(labels).mean().round(2))
            st.write("Average values per group. Compare the share column to see which group "
                     f"is most linked to '{pos}'.")
            st.dataframe(summ)

    with t4:
        st.subheader("Bias checker")
        st.write("Does the model work equally well for different groups in the data?")
        col = st.selectbox("Split rows by", list(X.columns), key="bias_col")
        if col in num_cols:
            cut = st.slider("Group boundary", float(X[col].min()), float(X[col].max()),
                            float(X[col].median()), key="bias_cut")
            groups = {f"{col} below {cut:.2f}": X[col] < cut, f"{col} at or above {cut:.2f}": X[col] >= cut}
        else:
            top = X[col].value_counts().index[:8]
            groups = {f"{col} = {v}": X[col] == v for v in top}
        rows = []
        for label, mask in groups.items():
            if mask.sum() == 0:
                continue
            act_f = (yy[mask] == fc)
            pr_f = (pred_idx_all[mask.values] == fc)
            rows.append({"Group": label, "Rows": int(mask.sum()),
                         f"Actually '{pos}' (%)": round(act_f.mean() * 100),
                         f"Predicted '{pos}' (%)": round(pr_f.mean() * 100),
                         "Accuracy": float((pred_idx_all[mask.values] == yy[mask].values).mean()),
                         f"Recall for '{pos}'": float((pr_f & act_f).sum() / act_f.sum()) if act_f.sum() else np.nan})
        bias = pd.DataFrame(rows).set_index("Group")
        st.dataframe(bias.style.format({"Accuracy": "{:.2f}", f"Recall for '{pos}'": "{:.2f}"}))
        acc_gap = bias["Accuracy"].max() - bias["Accuracy"].min() if len(bias) > 1 else 0
        if acc_gap > 0.15:
            st.error(f"⚠️ Accuracy differs by {acc_gap*100:.0f} points between groups, "
                     "so the model may treat some groups unfairly.")
        else:
            st.success("✅ Accuracy is similar across the groups shown.")
        st.caption("This check includes rows the model trained on, and small groups give unreliable numbers.")

        st.subheader("Ethics and limitations")
        st.markdown("""
- **A prediction is a pattern, not a fact.** Some predictions will be wrong, so use them to support decisions, not make them.
- **Data quality limits everything.** Small, biased or outdated data gives unreliable models.
- **Check fairness before trusting results** for any group of people.
- **Privacy:** only upload data you are allowed to use, and remove names and IDs.
- **What-if and "how to change" advice** shows what the model learned. It does not prove cause and effect.
""")

    with t5:
        st.subheader("The algorithms in this app")
        st.markdown("""
| Model | Lab experiment | In one line |
|---|---|---|
| Train/test split, k-fold CV | Exp 1 | Test on rows the model has never seen, and repeat several times for fairness |
| kNN | Exp 4 | Look at the k most similar rows and take a vote |
| Decision Tree | Exp 3 | Asks yes/no questions; too deep means it memorizes (overfitting) |
| Naive Bayes | Exp 7 | Uses probabilities, assuming features are independent |
| Logistic Regression | Exp 8 | Draws a straight boundary and gives a probability (sigmoid) |
| SVM | Exp 9 | Finds the widest-margin boundary; C and gamma control how tight it is |
| Neural Network | Exp 10 | Layers of neurons that learn curved patterns |
| K-Means | Exp 14 | Groups similar rows without being told the answer |
| ROC and AUC | Exp 8 | How well a model separates one class from the rest at every threshold |
""")

# =====================================================================
# 6. PREDICTOR MODE
# =====================================================================
else:
    st.subheader("Enter values for one case")
    with st.expander("Feature values (start at typical values)", expanded=True):
        me = input_widgets("me")
    p1, p2, p3 = st.tabs(["🎯 Prediction", "🎚️ What-if", "🛠️ How to change the outcome"])

    with p1:
        base, top = reasons_for(me)
        label = classes[int(model.predict(rows_df([me]))[0])] if multi else (pos if base >= 0.5 else "not " + str(pos))
        st.metric(f"Probability of '{pos}'", f"{base*100:.0f}%")
        st.write(f"**{winner}** predicts: **{label}**. This is a pattern-based estimate, not a certainty.")
        st.subheader("Why?")
        shown = False
        for c, val, avg, eff in top:
            if abs(eff) > 0.01:
                direction = "up" if eff > 0 else "down"
                arrow = "🔺" if eff > 0 else "🔻"
                st.write(f"{arrow} **{c}** = {val} (typical: {avg}) pushes the probability of '{pos}' "
                         f"**{direction}** by about {abs(eff)*100:.0f} points")
                shown = True
        if not shown:
            st.write("No single feature stands out for this case.")

    with p2:
        st.write("Change some features and see how the probability moves.")
        adjust = st.multiselect("Features to adjust", list(X.columns), default=list(X.columns)[:8])
        new = input_widgets("wi", start=me, shown=adjust, slider=True)
        before, after = prob_of(me), prob_of(new)
        m1, m2 = st.columns(2)
        m1.metric(f"Original probability of '{pos}'", f"{before*100:.0f}%")
        m2.metric("New probability", f"{after*100:.0f}%", delta=f"{(after-before)*100:.0f}%")

    with p3:
        want = st.selectbox("Desired outcome", classes,
                            index=0 if pos != classes[0] else min(1, len(classes) - 1))
        cur_pred = classes[int(model.predict(rows_df([me]))[0])] if multi else (pos if prob_of(me) >= 0.5 else [c for c in classes if c != pos][0])
        if str(cur_pred) == str(want):
            st.success(f"The model already predicts '{want}' for these values.")
        else:
            actionable = st.multiselect("Features that can be changed", list(X.columns),
                                        default=list(X.columns)[:8])
            if not actionable:
                st.info("Pick at least one feature that can be changed.")
            else:
                new, p0, pe, ok = find_change(me, actionable, want)
                ch = pd.DataFrame({"Now": {c: me[c] for c in actionable},
                                   "Suggested": {c: new[c] for c in actionable}})
                ch = ch[ch["Now"].astype(str) != ch["Suggested"].astype(str)]
                a, b = st.columns(2)
                a.metric(f"Probability of '{want}' now", f"{p0*100:.0f}%")
                b.metric("After the changes", f"{pe*100:.0f}%", delta=f"{(pe-p0)*100:.0f}%")
                if ok:
                    st.success(f"These changes make the model predict '{want}':")
                else:
                    st.warning("These features alone are not enough. Try adding more features to the list.")
                if len(ch):
                    st.dataframe(ch)
                st.caption("Suggestions come from patterns in the data, not a guaranteed outcome.")
