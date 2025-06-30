# task5_tree_models.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import graphviz

# Load the dataset
print("\n Loading the Heart Disease dataset...")

df = pd.read_csv("heart.csv")

print(" Dataset loaded successfully!\n")

# Show the first few rows
print(" First five rows:")
print(df.head())

# Info about the dataset
print("\n Dataset Info:")
print(df.info())

# Check for missing values
print("\n Missing values per column:")
print(df.isnull().sum())

# Prepare the data
X = df.drop("target", axis=1)
y = df["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Decision Tree model
print("\n Training Decision Tree Classifier...")
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

# Predict on test data
y_pred = tree_clf.predict(X_test)

# Evaluate the model
print("\n Decision Tree Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the tree using matplotlib
print("\n Visualizing the Decision Tree with Matplotlib...")
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_matplotlib.png")
plt.show()

# Export to Graphviz format for better visualization
print("\n Exporting Decision Tree to Graphviz format...")
dot_data = export_graphviz(
    tree_clf,
    out_file=None,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_graphviz", format="png", cleanup=True)

print("\n Decision tree visualizations saved as PNG files.")
