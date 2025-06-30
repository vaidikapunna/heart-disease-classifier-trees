# Task 5: Decision Trees and Random Forests

This task focuses on building and evaluating tree-based models — **Decision Trees** and **Random Forests** — using the **Heart Disease Dataset**. It is part of my AI/ML internship at Elevate Labs.
##  Folder Structure:
TASK 5/
├── task5_tree_models.py
├── task5_random_forest.py
├── heart.csv
└── README.md

---

 Objectives

- Load and explore the Heart Disease dataset
- Train a Decision Tree Classifier and evaluate its performance
- Visualize the Decision Tree using `Graphviz`
- Train a Random Forest Classifier and evaluate its performance
- Analyze feature importances

---

 Dataset

- **Source**: Heart Disease UCI Dataset
- **Attributes**: 14 (including age, sex, chest pain, etc.)
- **Target**: `0` (No heart disease), `1` (Presence of heart disease)
- **No. of rows**: 1025

---

 Files

### `task5_tree_models.py`

- Loads the dataset
- Performs data cleaning and checks for nulls
- Trains a Decision Tree model
- Prints evaluation metrics (accuracy, precision, recall, F1-score)
- Visualizes the decision tree using Matplotlib and `Graphviz`

### `task5_random_forest.py`

- Loads the same dataset
- Trains a Random Forest Classifier (100 trees)
- Evaluates model accuracy
- Displays feature importances using a horizontal bar chart

---

 Results

- **Decision Tree Accuracy**: `98.5%`
- **Random Forest Accuracy**: `96.5%`
- Models are well-fitted and evaluated on unseen test data.

---

 Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- graphviz (installed and PATH configured for visualization)

 Notes

- `graphviz` was downloaded from the official site and manually added to system PATH to enable visualization.
- Both scripts run independently.
- Feature importance from Random Forest helps understand model decisions.

---

 Conclusion

This task provided hands-on experience with decision tree logic and ensemble learning. It also strengthened my skills in data preprocessing, visualization, and model evaluation.


