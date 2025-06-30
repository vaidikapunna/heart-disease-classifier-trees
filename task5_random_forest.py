# task5_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
print("\n Loading the Heart Disease dataset...")
df = pd.read_csv("heart.csv")
print("Dataset loaded successfully!\n")
print("First five rows:")
print(df.head(), "\n")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Training Random Forest Classifier...\n")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(" Random Forest Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
plt.figure(figsize=(12, 6))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', color='forestgreen')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
