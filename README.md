# ML Model Arena 🏟️

Upload **any CSV**, choose the column to predict, and six machine learning models
compete on it. The app picks the fairest winner, explains the result in plain
language, and lets you test what-if changes.

**Live app:** <**Live app:** https://mlmodelarena-qr8aokjkhvz6drtsmv25up.streamlit.app/>

## Works on any classification dataset
- Handles **numeric and text columns**, missing values, and ID-like columns (ignored automatically).
- Works with **2 classes or more** (for example Safe/At Risk, spam/ham, or three flower species).
- Built-in demos: Breast cancer (2 classes, 30 features) and Iris (3 classes), both from scikit-learn, plus a simulated **space-health** demo (3 risk levels, mixed numeric and text columns) linked to my seminar on AI-guided ultrasound in space missions.
- Limit: the column to predict must be a category with up to 20 labels (classification, not regression).

## What it does
| Feature | Lab experiment |
|---|---|
| 80/20 train-test split and k-fold cross-validation | Exp 1 |
| Decision tree with depth control, tree picture and overfitting alarm | Exp 3 |
| k-Nearest Neighbours with adjustable k | Exp 4 |
| Gaussian Naive Bayes | Exp 7 |
| Logistic Regression, ROC curves and AUC | Exp 8 |
| SVM with kernel, C and gamma controls | Exp 9 |
| Neural network (MLP) with activation and layer controls | Exp 10 |
| K-Means groups with elbow and silhouette charts | Exp 14 |

Two modes:
- **Analyst:** scoreboard and winner, overfitting alarm, ROC curves, feature importance, decision tree, K-Means groups, bias checker.
- **Predictor:** enter one case to get a prediction with reasons, a What-if simulator, and a "how to change the outcome" search (counterfactual advice).

## Data note
The breast cancer and Iris demos come with scikit-learn and are real, public datasets. The space-health demo is **simulated**: no public astronaut dataset exists, so it is generated from published figures (about 1 to 1.5% bone loss per month in space, reduced by exercise), and it is not real crew data. The breast cancer
demo is for learning only and must not be used for medical decisions.

## Why F1 and not accuracy
When one class is rare, a model that always predicts the common class still scores a
high accuracy. The app ranks models by cross-validated F1 (macro F1 for more than two
classes) and uses `class_weight="balanced"` where the model supports it.

## Run it yourself
```
pip install -r requirements.txt
streamlit run app.py
```
Files: `app.py` (interface) and `engine.py` (data cleaning and models).

## Limitations and ethics
- Predictions are patterns, not certainty, and the demo is not medical advice. Use them to support decisions, not make them.
- What-if and "how to change" advice shows what the model learned, not cause and effect.
- Check the bias checker before trusting results for any group of people.
- Only upload data you are allowed to use, and remove names and IDs first.

## Tech
Python, Streamlit, scikit-learn, pandas, matplotlib.


Built by < KOUNSALYE SIDDHI ARUN >,Roll No:216 , [T5] TYCSE-B, MGM's College of Engineering, Nanded.
