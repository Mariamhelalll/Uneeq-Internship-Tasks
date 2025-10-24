# Uneeq-Internship-Tasks
ML Internship Tasks

# 🧠 Customer Churn Prediction  
### Uneeq Interns — Task 2  

---

## 🎯 Project Objective  
The aim of this project is to develop a **machine learning model** that predicts whether a customer will **churn** (cancel their subscription) from a subscription-based business.  

The project focuses on:  
- Handling **imbalanced datasets**  
- Exploring multiple **classification algorithms**  
- Evaluating using **precision, recall, and ROC-AUC**  
- Deriving **business insights** that help reduce churn and improve customer retention  

---

## 📊 Dataset Overview  

The dataset includes customer demographic, usage, financial, and support interaction information.  

| Feature Category | Description | Example Features |
|------------------|--------------|------------------|
| Demographics | Basic customer attributes | Age, Tenure |
| Usage Patterns | Service usage behaviour | Usage Frequency, Last Interaction |
| Support Metrics | Interaction with customer service | Support Calls |
| Financial Data | Spending and payment behaviour | Payment Delay, Total Spend |
| Target Variable | Churn status | `Churn` (1 = Yes, 0 = No) |

---

## 🧹 Data Preprocessing  

1. **Missing Values**  
   - Numeric → Replaced with median  
   - Categorical → Replaced with mode  

2. **Data Cleaning**  
   - Converted object-type numerics (e.g., TotalCharges) to numeric  
   - Dropped non-informative columns like `CustomerID`  

3. **Encoding & Scaling**  
   - Label encoded categorical columns  
   - Standardised numerical features for model training  

4. **Outlier Handling**  
   - Removed extreme outliers using the IQR method  

---

## 📈 Exploratory Data Analysis (EDA)  

### 🔍 1. Data Overview  
- Verified data types, missing values, and duplicates  
- Identified categorical vs numerical variables  

### 📊 2. Univariate Analysis  
- **Numerical:** Histograms + boxplots for distribution and outliers  
- **Categorical:** Countplots (with `hue='Churn'`) for churn distribution  

### 🔗 3. Bivariate Analysis  
- **Contract Type:** Month-to-month users showed highest churn rate  
- **Support Calls:** High support interaction correlated with churn  
- **Payment Delay:** Frequent delays linked to increased churn probability  

### 🔥 4. Correlation Analysis (Numerical)  
- Created a correlation heatmap to identify relationships  

**Key Correlations:**  
- `Support Calls` → Churn (**r = 0.57**)  
- `Payment Delay` → Churn (**r = 0.31**)  
- `Total Spend` → Churn (**r = -0.43**)  
- `CustomerID` → Irrelevant correlation (removed before modelling)  

### 🧩 5. Categorical Association (Cramer’s V)  
- Strong relationships found for `Contract`, `InternetService`, and `TechSupport`.  

---

## ⚖️ Handling Class Imbalance  
Churn datasets are typically imbalanced (few churners).  
- Used **SMOTE (Synthetic Minority Oversampling Technique)** on the training set only.  
- Ensured no synthetic data leakage into test data.  

---

## 🤖 Model Building  

| Model | Description | Purpose |
|--------|--------------|----------|
| **Logistic Regression** | Simple linear baseline model | Benchmark performance |
| **Random Forest** | Ensemble model | Captures non-linear relationships & improves recall |

---

## 📊 Model Evaluation  

**Metrics Used:**  
- Precision, Recall, F1-Score  
- ROC-AUC  
- Confusion Matrix  

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 81.2% | 88.9% |
| Precision (Churn) | 0.71 | 0.82 |
| Recall (Churn) | 0.64 | 0.87 |
| F1-Score | 0.67 | 0.84 |
| ROC-AUC | 0.79 | 0.91 |

✅ **Result:** Random Forest achieved the best balance between recall and precision, making it ideal for identifying customers likely to churn.

---

## 🔍 Feature Importance (Random Forest)

Top features influencing churn:  
1. Support Calls  
2. Contract Type  
3. Payment Delay  
4. Total Spend  
5. Internet Service  
6. Tenure  
7. Tech Support  
8. Payment Method  
9. Monthly Charges  
10. Last Interaction  

---

## 💼 Business Insights & Recommendations  

| Insight | Recommended Action |
|----------|--------------------|
| High support call volume correlates with churn | Improve customer service and response quality |
| Frequent payment delays precede churn | Offer reminders or flexible payment options |
| Low spenders tend to churn more | Introduce loyalty or discount programs |
| Month-to-month users churn most | Incentivize long-term contracts |
| Lack of tech support increases churn | Offer affordable tech support plans |

> 💡 **Summary:** Customers who are frustrated (many support calls), financially constrained (delayed payments), or not locked into contracts are most likely to churn.  

---

## 🧠 Key Learnings  
- Precision and recall provide a better understanding of model quality than accuracy alone.  
- Balancing datasets using SMOTE improves model generalisation.  
- Combining technical analysis with business interpretation is essential in real-world data science.  

---

## 🧩 Tech Stack  

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python |
| Analysis & Visualisation | Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Environment | Google Colab / Jupyter |
| Version Control | GitHub |

---

## 🧾 Project Workflow 
1️⃣ Data Loading
2️⃣ Data Cleaning & Preprocessing
3️⃣ Exploratory Data Analysis (EDA)
4️⃣ Feature Engineering
5️⃣ Model Training (Baseline + Advanced)
6️⃣ Evaluation & Comparison
7️⃣ Insights & Recommendations


---

## 📦 Deliverables  

- 📒 **Notebook:** [Uneeq Interns Task 2 — Customer Churn Prediction.ipynb](Uneeq_interns_Task_2_Customer_Churn_Prediction.ipynb)  
- 🧮 **Code Repository:** Public GitHub Repo  
- 🎥 **Video Explanation:** Uploaded to YouTube (demonstrating workflow & results)  
- 💬 **LinkedIn Post:** Summary post tagging **Uneeq Interns**  

---

## 🏆 Final Summary  

This project demonstrates both **technical and analytical excellence**:  
- A complete ML workflow from raw data → insights → predictive model.  
- Balanced focus on **business relevance** and **statistical accuracy**.  
- Clear visual storytelling through EDA and model interpretation.  

> ✨ **By identifying churn drivers early, the business can proactively retain customers and reduce revenue loss.**

---

## 👩‍💻 Author  

**Maryam Mohamed**  
Uneeq Internship (ML Track)  

📍 **LinkedIn:** [Your LinkedIn URL]  
🧭 **GitHub:** [Your GitHub Profile]  
📧 **Email:** [Your Email Address]

---


