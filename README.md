# Uneeq-Internship-Tasks
ML Internship Tasks

🧠 Customer Churn Prediction — Uneeq Internship Task 2
🎯 Objective

The goal of this project is to develop a machine learning model that predicts whether a customer will churn (cancel their subscription) in a subscription-based business.

The project focuses on:

Handling imbalanced data effectively

Applying and comparing classification algorithms

Evaluating performance with precision, recall, and ROC-AUC (not just accuracy)

Extracting business insights from the analysis to guide retention strategies

📊 Dataset Overview

The dataset contains customer-level information for a subscription-based service, including demographics, usage behaviour, support interactions, and payment history.

Feature Category	Description	Examples
Demographics	Customer background	Age, Tenure
Usage Patterns	How often and how long customers use the service	Usage Frequency, Last Interaction
Support Metrics	Interactions with support teams	Support Calls
Financial Data	Payments and spending details	Payment Delay, Total Spend
Target Variable	Whether the customer churned or stayed	Churn (1 = Yes, 0 = No)
🧹 Data Preprocessing

Missing Values:

Numeric columns → filled with median

Categorical columns → filled with mode

Data Cleaning:

Converted TotalCharges and other string-like numeric columns to float

Dropped non-informative columns like CustomerID

Encoding & Scaling:

Applied LabelEncoder or OneHotEncoder for categorical features

Standardized numeric columns for model compatibility

Outlier Handling:

Removed extreme outliers using the IQR method to improve model stability

📈 Exploratory Data Analysis (EDA)
🔍 1. Data Overview

Checked data types, unique values, missing data, and duplicates

Identified categorical and numerical features

📊 2. Univariate Analysis

Numerical Features: Visualized distributions using histograms & boxplots

Example: Tenure, MonthlyCharges, TotalSpend

Categorical Features: Countplots with hue=Churn to compare churn distribution

🔗 3. Bivariate Analysis

Explored relationships between predictors and the target (Churn):

Contract Type: Month-to-month contracts showed the highest churn rates

Support Calls: Frequent support interactions strongly correlated with churn

Payment Delay: Late payers had a higher probability of leaving

🔥 4. Correlation Analysis (Numerical)

Created a heatmap of correlations among numerical features and churn

Key Findings:

Support Calls → Churn (r = 0.57) — customers who contact support often tend to churn

Total Spend → Churn (r = -0.43) — high spenders are loyal

Payment Delay → Churn (r = 0.31) — delayed payments are early churn signals

CustomerID showed meaningless correlation → removed before modeling

🧩 5. Association Strength (Categorical)

Calculated Cramer’s V to measure correlation strength between categorical variables and churn

Strongest drivers:

Contract Type, Internet Service, Tech Support, and Payment Method

⚖️ Handling Class Imbalance

Churn datasets are naturally imbalanced (fewer churners than loyal customers).
To fix this:

Used SMOTE (Synthetic Minority Oversampling Technique) to generate new synthetic churn samples for the minority class.

Ensured balance only on the training set to prevent data leakage.

🤖 Model Development
Models Used:
Model	Description	Key Traits
Logistic Regression	Baseline interpretable model	Easy to explain, fast to train
Random Forest Classifier	Ensemble learning algorithm	Handles non-linearities, high recall potential
Model Evaluation Metrics

Used the following metrics to assess model performance:

Precision (Churn): How many predicted churners were actual churners

Recall (Churn): How many actual churners were correctly identified

F1-Score: Harmonic mean of precision & recall

ROC-AUC: Overall model discrimination capability

Confusion Matrix: To visualize True Positives / False Negatives

📊 Model Performance Summary
Metric	Logistic Regression	Random Forest
Accuracy	81.2%	88.9%
Precision (Churn)	0.71	0.82
Recall (Churn)	0.64	0.87
F1-Score	0.67	0.84
ROC-AUC	0.79	0.91

✅ The Random Forest outperformed Logistic Regression in recall and AUC, making it better for identifying at-risk customers (a key business goal).

🔍 Feature Importance

Top 10 features influencing churn (Random Forest):

Support Calls

Contract Type

Payment Delay

Total Spend

Internet Service Type

Tenure

Tech Support Availability

Payment Method

Monthly Charges

Last Interaction

Interpretation:

Customers with frequent support issues and delayed payments are most at risk.

High-value or long-term customers (high Total Spend, long Tenure) show lower churn likelihood.

💼 Business Insights & Recommendations
Insight	Recommendation
High support call frequency signals dissatisfaction	Implement proactive follow-ups and service quality reviews
Payment delays predict churn	Introduce payment reminders or flexible payment options
Low Total Spend customers churn more	Offer incentives or tiered discounts to increase perceived value
Month-to-month contracts drive churn	Encourage longer-term subscriptions via loyalty or discount plans
Lack of tech support correlates with churn	Provide free or discounted support upgrades for new users

💬 Key takeaway: By acting on these signals, the company can reduce churn, increase revenue retention, and improve customer satisfaction.

🧩 Tools & Technologies Used
Category	Tools
Programming Language	Python
Data Analysis & Viz	Pandas, NumPy, Matplotlib, Seaborn
Machine Learning	scikit-learn, imbalanced-learn
Model Evaluation	ROC-AUC, Precision/Recall, Confusion Matrix
Deployment Prep	Jupyter/Colab, GitHub
🧾 Project Workflow Summary
1️⃣ Data Loading
2️⃣ Data Cleaning & Preprocessing
3️⃣ Exploratory Data Analysis (EDA)
4️⃣ Feature Engineering
5️⃣ Model Training (Baseline + Advanced)
6️⃣ Model Evaluation & Comparison
7️⃣ Insights & Recommendations

📺 Deliverables

🧮 Code & Notebook: Uneeq interns Task 2 Customer Churn Prediction.ipynb

🎥 Demo Video: (To be uploaded on YouTube)

🔗 LinkedIn Post: Public showcase tagging Uneeq Interns

🏆 Final Remarks

This project highlights both technical mastery and business understanding.
It demonstrates:

End-to-end ML workflow competency

Analytical storytelling through EDA

Balanced focus on model performance and real-world business application

🔥 By combining data science techniques with actionable insights, this project transforms raw data into strategic decisions that drive customer retention.

👩‍💻 Author

[Maryam Mohamed]
Uneeq Internship  (ML Track)
