"""Dataset-agnostic ML engine (no Streamlit inside, so it can be tested alone)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


# ---------------------------------------------------------------- cleaning
def clean(df, target, max_rows=3000):
    """Return X (raw features), y, classes, num_cols, cat_cols, notes."""
    notes = []
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.dropna(subset=[target])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
        notes.append(f"Large file: a random sample of {max_rows} rows is used to keep the app fast.")
    df = df.reset_index(drop=True)

    X = df.drop(columns=[target])
    y = df[target]

    # text columns that are really numbers (e.g. "75") become numeric
    for c in X.columns:
        if X[c].dtype == object:
            conv = pd.to_numeric(X[c], errors="coerce")
            if conv.notna().sum() >= 0.9 * X[c].notna().sum() and X[c].notna().sum() > 0:
                X[c] = conv

    dropped = []
    for c in list(X.columns):
        s = X[c]
        n = s.nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if n <= 1:
            dropped.append(f"{c} (constant)")
        elif pd.api.types.is_datetime64_any_dtype(s):
            dropped.append(f"{c} (date)")
        elif n == len(s) and not pd.api.types.is_float_dtype(s):
            dropped.append(f"{c} (looks like an ID)")
        elif not is_num and n > 30:
            dropped.append(f"{c} (too many different text values)")
        else:
            continue
        X = X.drop(columns=c)
    if dropped:
        notes.append("Columns ignored: " + ", ".join(dropped) + ".")

    num_cols = [c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:                       # make text columns uniform strings
        s = X[c]
        X[c] = s.where(s.isna(), s.astype(str)).astype(object)

    miss = int(X.isna().sum().sum())
    if miss:
        notes.append(f"{miss} missing values were filled automatically "
                     "(median for numbers, most common value for text).")

    classes = sorted([c.item() if hasattr(c, 'item') else c for c in y.unique()], key=lambda v: str(v))
    return X, y, classes, num_cols, cat_cols, notes


# ---------------------------------------------------------------- models
def make_pre(num_cols, cat_cols, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    parts = []
    if num_cols:
        parts.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    return ColumnTransformer(parts)


def build_models(num_cols, cat_cols, p):
    def pipe(clf, scale=True):
        return Pipeline([("pre", make_pre(num_cols, cat_cols, scale)), ("clf", clf)])
    return {
        "kNN": pipe(KNeighborsClassifier(n_neighbors=p["k"])),
        "Decision Tree": pipe(DecisionTreeClassifier(
            max_depth=p["depth"], random_state=42, class_weight="balanced"), scale=False),
        "Naive Bayes": pipe(GaussianNB(), scale=False),
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM": pipe(SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"], probability=True,
                        random_state=42, class_weight="balanced")),
        "Neural Network": pipe(MLPClassifier(
            hidden_layer_sizes=(p["h1"], p["h2"]), activation=p["act"],
            max_iter=1000, random_state=42)),
    }


def encode_target(y, classes, pos):
    """Binary: 1 for the chosen class, else 0.  Multi-class: index of the class."""
    if len(classes) == 2:
        return (y == pos).astype(int)
    idx = {c: i for i, c in enumerate(classes)}
    return y.map(idx).astype(int)


def focus_column(classes, pos):
    """Column of predict_proba that belongs to the class of interest."""
    return 1 if len(classes) == 2 else classes.index(pos)


def train_all(X, y, classes, pos, num_cols, cat_cols, p):
    multi = len(classes) > 2
    yy = encode_target(y, classes, pos)
    min_count = int(yy.value_counts().min())
    if min_count < 2:
        raise ValueError("Every class needs at least 2 rows.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, yy, test_size=0.2, random_state=42, stratify=yy)             # Exp 1
    avg = "macro" if multi else "binary"
    scoring = "f1_macro" if multi else "f1"
    skf = StratifiedKFold(n_splits=max(2, min(5, min_count)), shuffle=True, random_state=42)

    models = build_models(num_cols, cat_cols, p)
    rows, probas, gaps = [], {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        prob = m.predict_proba(X_test)
        probas[name] = prob
        gaps[name] = accuracy_score(y_train, m.predict(X_train)) - accuracy_score(y_test, pred)
        try:
            auc = (roc_auc_score(y_test, prob[:, 1]) if not multi
                   else roc_auc_score(y_test, prob, multi_class="ovr", average="macro"))
        except ValueError:
            auc = np.nan
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
            "F1": f1_score(y_test, pred, average=avg, zero_division=0),
            "AUC": auc,
            "CV F1": cross_val_score(m, X, yy, cv=skf, scoring=scoring).mean(),
        })
    score = pd.DataFrame(rows).set_index("Model")
    return dict(models=models, score=score, probas=probas, gaps=gaps,
                X_test=X_test, y_test=y_test, yy=yy, multi=multi, scoring=scoring,
                folds=skf.get_n_splits(), winner=score["CV F1"].idxmax())
