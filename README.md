# Titanic Survival Prediction

Simple machine learning workflow predicting passenger survival on the Titanic using standard scikit-learn models. Built as a hands-on project to practice data cleaning, feature handling, and basic classification.

## Tech Stack

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- Jupyter Notebook

## What Was Done

1. **Handling Missing Values:**
   - Filled missing `Age` values using median age grouped across classes and sex.
   - Dropped `Cabin` due to high sparsity (~77% nulls).
   - Filled missing `Embarked` entries with the mode (`S`).

2. **Feature Engineering:**
   - Extracted honorific titles (`Mr`, `Miss`, `Mrs`, `Master`) from `Name`.
   - Combined `SibSp` and `Parch` into a single `FamilySize` feature.
   - Converted categorical variables (`Sex`, `Embarked`) via one-hot/label encoding.
   - Dropped non-predictive identifiers like `PassengerId` and `Ticket`.

3. **Models Tested:**
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest
   - Support Vector Classifier (SVC)

## Quickstart

```bash
# Clone repository
git clone [https://github.com/PremNikam1402/Titanic-survival-prediction-ML-project.git](https://github.com/PremNikam1402/Titanic-survival-prediction-ML-project.git)
cd Titanic-survival-prediction-ML-project

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
