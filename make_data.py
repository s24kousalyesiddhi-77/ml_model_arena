import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

attendance = np.random.normal(75, 15, n).clip(30, 100)
study_hours = np.random.normal(10, 4, n).clip(0, 25)
internal_marks = (attendance * 0.25 + study_hours * 1.2 + np.random.normal(0, 6, n)).clip(0, 40)
prev_cgpa = np.random.normal(7, 1.2, n).clip(4, 10)
assignments = np.random.randint(40, 101, n)
sleep_hours = np.random.normal(6.5, 1.2, n).clip(3, 10)

z = (0.05 * (attendance - 70) + 0.1 * (study_hours - 10) + 0.1 * (internal_marks - 20)
     + 0.6 * (prev_cgpa - 7) + 0.02 * (assignments - 70) + np.random.normal(0, 0.5, n))

result = np.where(z < -0.8, "At Risk", "Safe")

df = pd.DataFrame({
    "attendance": attendance.round(1),
    "study_hours": study_hours.round(1),
    "internal_marks": internal_marks.round(1),
    "prev_cgpa": prev_cgpa.round(2),
    "assignments_pct": assignments,
    "sleep_hours": sleep_hours.round(1),
    "result": result,
})
df.to_csv("students.csv", index=False)
print(df["result"].value_counts())