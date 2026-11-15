# EduGuard 🎓 — an ML Model Arena for finding at-risk students early

EduGuard lets several machine learning models compete on student data, picks the
fairest winner, and then explains the result in plain language: who is at risk,
why, and what could change it.

**Live app:** <paste your Streamlit link here>

![EduGuard screenshot](screenshots/scoreboard.png)

## Data note (please read)
The included `students.csv` is **synthetic data** created by `make_data.py`
(300 students, 6 features). It is meant for demonstration, so the results are
not findings about real students. The app accepts any CSV with a two-class
target column, so real data can be uploaded from the sidebar.

## What it does
| Feature | Lab experiment |
|---|---|
| 80/20 train-test split and 5-fold cross-validation | Exp 1 |
| Linear and kernel models compared on one scoreboard | Exp 1, 9 |
| Decision tree with depth control, tree picture and overfitting alarm | Exp 3 |
| k-Nearest Neighbours with adjustable k | Exp 4 |
| Gaussian Naive Bayes | Exp 7 |
| Logistic Regression, ROC curves and AUC | Exp 8 |
| SVM with kernel, C and gamma controls | Exp 9 |
| Neural network (MLP) with activation and layer controls | Exp 10 |
| K-Means student groups with elbow and silhouette charts | Exp 14 |

Extras: What-If simulator, "Why this prediction?" (permutation importance and
per-student reasons), counterfactual advice ("how to fix it"), a bias checker,
a Data Doctor for missing values, and separate Teacher and Student views.

## Results on the sample data
Fill this in from your own scoreboard:

| Model | Accuracy | Recall | F1 | AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|
| ... | | | | | |

**Winner:** <model name>. 
**Why F1, not accuracy:** only about 20% of students are At Risk, so a model
that says "Safe" for everyone would still score about 80% accuracy.
`class_weight="balanced"` raised recall for Logistic Regression and SVM.

## Run it yourself
```
pip install -r requirements.txt
python make_data.py
streamlit run app.py
```

## Limitations and ethics
- Synthetic data, so conclusions do not transfer to real students.
- A flag means "offer help", never "punish". Predictions are patterns, not certainty.
- Counterfactual advice shows what the model learned, not a guaranteed outcome.
- Always check the Bias checker before trusting results for any group.

## Tech
Python, Streamlit, scikit-learn, pandas, matplotlib.

Built by <your name>, TYCSE, MGM's College of Engineering, Nanded.
