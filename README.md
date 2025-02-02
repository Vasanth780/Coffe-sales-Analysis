# Coffe-sales-Analysis
**Coffee Sales Data Analysis**

This project performs a comprehensive analysis of coffee sales data, covering steps from data cleaning and exploration to building a machine learning model that predicts sales based on various features.

Table of Contents
Introduction
Dataset Description
Setup
Data Preparation and Cleaning
Handling Missing Values
Converting Data Types
Removing Outliers
Exploratory Data Analysis (EDA)
Sales Over Time
Sales by Coffee Type
Machine Learning Modeling
Feature Engineering
Splitting the Data
Training the Model
Model Evaluation
Model Interpretation
Conclusion

**Introduction**
This project analyzes coffee sales data, cleans and prepares the data, performs exploratory analysis, and finally builds a simple linear regression model to predict sales. The objective is to uncover trends and understand the factors affecting sales.

**Dataset Description**
The dataset includes coffee sales transactions with the following columns:

date: The transaction date.
datetime: Full timestamp of the transaction.
cash_type: Payment method (e.g., card).
card: An anonymized identifier for the payment card.
money: Amount spent in the transaction (target variable).
coffee_name: The type of coffee purchased.

**Setup**

**Clone the repository:**

bash
Copy code
git clone https://github.com/Vasanth780
Navigate to the project directory:

bash
Copy code
cd coffee-sales-analysis
Install the required Python libraries:

bash
Copy code
pip install pandas scikit-learn matplotlib seaborn
Place the coffee_sales.csv file in the project directory.

**Run the Python script to execute the analysis:
**
bash
Copy code
python coffee_sales_analysis.py
Data Preparation and Cleaning

**Handling Missing Values:**

Missing numerical values (money) are filled using the median value.
Categorical values (coffee_name) are filled with the most frequent (mode) value.
Converting Data Types:

The date column is converted to a datetime format to allow for time-based operations.

**Removing Outliers:**

Outliers in the numerical columns (money) are removed using Z-scores with a threshold of 3 standard deviations.
Exploratory Data Analysis (EDA)
Visualizations are generated to uncover patterns and insights:

**Sales Over Time:**

A line plot showing how sales vary over different months of the year.
Example Code:

python
Copy code
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Month', y='money', hue='Year')
plt.title('Monthly Sales Over Years')
plt.show()
Sales by Coffee Type:

A bar plot that visualizes which coffee types contribute the most to sales.
Example Code:

python
Copy code
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='coffee_name', y='money')
plt.title('Sales by Coffee Type')
plt.show()

**Machine Learning Modeling**

**Feature Engineering**

Extract the month and year from the date column and use them as features.
Drop the original date column after extraction.

**Splitting the Data**

The dataset is split into training and testing sets:

python
Copy code
from sklearn.model_selection import train_test_split

X = data.drop(columns=['money'])
y = data['money']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Training the Model
A simple linear regression model is trained on the dataset:

python
Copy code
from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
Model Evaluation
Evaluate the model using Mean Squared Error (MSE) and R-squared (R²) score:

python
Copy code
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

**Model Interpretation**
You can interpret the model by looking at the coefficients for each feature:

python
Copy code
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
Conclusion
This project demonstrates the full pipeline from data cleaning and feature engineering to building and evaluating a machine learning model for sales prediction. The analysis highlights key factors influencing coffee sales and offers insights for further business decision-making.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
If you have any questions, feel free to reach out via vasanth780.com or create an issue in the GitHub repository.

