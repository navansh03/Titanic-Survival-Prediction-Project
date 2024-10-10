
# **Titanic Survival Prediction Project**

This repository contains a machine learning project that aims to predict the survival of passengers aboard the Titanic using various machine learning models. The project is based on the famous Titanic dataset, part of a Kaggle competition.

## **Overview**
This project explores the Titanic dataset and builds a prediction model to estimate whether a passenger survived or not, based on features such as gender, age, ticket class, etc.

### **Key Achievements**:
- Achieved **76.55% accuracy** (Top 12%) in the Kaggle Titanic competition.
  
### **Project Structure**:
1. **Understanding Data** (Exploration through visualizations like histograms, box plots).
2. **Data Cleaning** (Handling missing values, feature transformation).
3. **Feature Engineering** (Creating new features).
4. **Preprocessing for Model** (Scaling, encoding categorical variables).
5. **Basic Model Building** (Comparison of different machine learning models).
6. **Model Tuning** (Hyperparameter tuning using GridSearchCV).
7. **Ensemble Modeling** (Combining multiple models to boost performance).
8. **Final Result**.

## **Dataset**
The dataset used for this project is the **Titanic dataset** available from [Kaggle](https://www.kaggle.com/c/titanic/data). It contains the following files:
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `gender_submission.csv`: Example of submission format for Kaggle.

### **Features**:
- **Survived**: Outcome (1 = Survived, 0 = Did not survive).
- **Pclass**: Passenger’s class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Name, Sex, Age**: Demographic details.
- **SibSp, Parch**: Family details (Siblings/Spouses, Parents/Children).
- **Ticket, Fare, Cabin**: Ticket and travel details.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### **Data Preprocessing**:
- **Missing Values**: Imputation strategies were applied for missing values in features like `Age`, `Fare`, `Cabin`, and `Embarked`.
- **Feature Engineering**: Created new features such as `cabin_multiple`, `wealth_indicator`, and `family_size`.

## **Modeling Approach**

### **Basic Models**:
Several basic models were trained and evaluated, including:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Classifier (SVC)**
- **XGBoost Classifier**

### **Model Tuning**:
Hyperparameter tuning was done using **GridSearchCV** for better performance.

### **Ensemble Modeling**:
An ensemble model was built by combining the best-performing individual models to improve the accuracy and generalization of predictions.

## **Model Evaluation**
- **Accuracy**: The model achieved an accuracy of **76.55%** on the test data.
- **Cross-Validation**: Models were validated using k-fold cross-validation (CV = 5) to ensure robustness.
- **Confusion Matrix**: Used to evaluate the precision, recall, and F1 score for the final model.

## **How to Use**

### **Installation**:
1. Clone the repository:
    ```bash
    git clone https://github.com/[YourUsername]/titanic-prediction.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### **Data**:
Place the data files (`train.csv`, `test.csv`, etc.) in the `./Data/` directory.

### **Running the Code**:
- To preprocess the data and train the model:
    ```bash
    python src/train.py
    ```

- To make predictions on the test set:
    ```bash
    python src/predict.py
    ```

### **Notebooks**:
For detailed analysis and exploratory data analysis (EDA), check the Jupyter notebooks in the `notebooks/` directory:
- `Titanic_Project.ipynb`

## **Dependencies**
The project was developed using the following Python libraries:
- **pandas**: Data manipulation and analysis.
- **numpy**: Array operations.
- **scikit-learn**: Machine learning models and utilities.
- **xgboost**: Extreme gradient boosting.
- **seaborn**: Statistical data visualization.
- **matplotlib**: Plotting and visualizing data.

You can install all the dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## **Results**
The final model's performance on the test set achieved **76.55% accuracy**, placing it in the **Top 12%** of the Titanic Kaggle competition leaderboard.

## **Contributing**
Feel free to fork this repository and contribute by making a pull request. For major changes, please open an issue first to discuss what you would like to change.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
