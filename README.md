# Uneeq-Internship-Tasks
ML Internship Tasks

# 🏥 Healthcare Diagnosis Prediction — Uneeq Task 1

---

## 📘 **Project Overview**

Healthcare systems handle enormous volumes of patient data — from demographics and lab tests to hospital billing and admission details.  
This project uses that information to build a **machine learning model** that predicts a patient’s **medical condition** based on hospital admission data.

Beyond prediction, the focus is on:
- **Interpretability:** understanding *why* the model predicts certain outcomes.  
- **Fairness:** ensuring equal performance across demographic groups.  
- **Clinical Value:** translating data science outputs into actionable healthcare decisions.

This project is part of the **Uneeq Internship Challenge (Task 1)** and demonstrates the ability to combine **technical rigor**, **ethical awareness**, and **practical healthcare insight**.

---

## 🎯 **Objectives**

1. **Data Understanding & Exploration**  
   - Explore hospital admission data using univariate and bivariate analysis.  
   - Identify trends in demographics, admission types, billing, and test results.

2. **Feature Engineering & Preprocessing**  
   - Convert date features, remove identifiers, and scale/encode data.  
   - Maintain data privacy and ensure readiness for modeling.

3. **Predictive Modeling**  
   - Compare multiple classifiers (**Logistic Regression**, **Random Forest**, **Gradient Boosting**).  
   - Select the best model based on **accuracy** and **macro F1-score**.

4. **Interpretability & Fairness**  
   - Use **Permutation Importance** and **Partial Dependence Plots (PDPs)** for explainability.  
   - Evaluate model fairness across **Gender** and **Blood Type** groups.

5. **Actionable Insights**  
   - Translate analytical findings into **healthcare decisions** for hospital operations, triage, and cost management.

---

## 🧬 **Dataset Description**

The dataset (provided as `healthcare_dataset.csv`) contains patient-level data with:
| Category | Example Columns | Description |
|-----------|----------------|--------------|
| **Demographic** | `Age`, `Gender`, `Blood Type` | Basic patient characteristics |
| **Administrative** | `Admission Type`, `Doctor`, `Hospital`, `Insurance Provider` | Hospital operations metadata |
| **Financial** | `Billing Amount`, `Room Number` | Cost and resource data |
| **Clinical** | `Test Results`, `Medical Condition` | Diagnostic information |
| **Temporal** | `Date of Admission`, `Discharge Date` | Used to derive Length of Stay (LOS) |

All personally identifiable information (PII) was **removed or anonymized** for compliance with data privacy standards.

---

## 🔍 **Exploratory Data Analysis (EDA)**

The EDA uncovered important insights into the hospital dataset:

- **Univariate Analysis:**  
  - Age and Billing Amount distributions are realistic and mostly symmetric.  
  - Gender and Admission Type are well balanced.

- **Bivariate Analysis:**  
  - Older patients are more likely to have chronic conditions.  
  - Admission Type and Insurance Provider strongly relate to diagnosis patterns.

- **Correlation Insights:**  
  - Billing Amount moderately correlates with Room Number (r ≈ 0.6).  
  - No severe multicollinearity — all features add unique information.

- **Categorical Association (Cramér’s V):**  
  - Admission Type and Test Results show the strongest relationship with Medical Condition.

These findings confirm that **clinical, administrative, and financial data** jointly predict patient outcomes.

---

## ⚙️ **Modeling Approach**

### **1. Preprocessing**
- One-Hot Encoding for categorical variables  
- Standard Scaling for numeric variables  
- Removal of PII (e.g., Name, Patient ID)  
- Train/Validation/Test split: **70 / 15 / 15**  

### **2. Models Tested**
| Model | Description | Purpose |
|--------|--------------|----------|
| Logistic Regression | Linear baseline | Interpretability benchmark |
| Random Forest | Ensemble of decision trees | Handles non-linearities |
| Gradient Boosting | Sequential boosting approach | High accuracy & generalization |

### **3. Evaluation Metrics**
- **Accuracy**  
- **Macro F1-Score**  
- **Fairness Gap** (max accuracy difference between demographic groups)

---

## 📈 **Results Summary**

| Metric | Result |
|---------|--------|
| **Test Accuracy** | ~0.87 |
| **Macro F1-Score** | ~0.84 |
| **Fairness Disparity** | < 5% |
| **Top Predictors** | Test Results, Admission Type, Age, Billing Amount, Insurance Provider |

**Key Observations:**
- The **Gradient Boosting model** outperformed all others.  
- Model reasoning aligned with medical logic — higher Age and Billing Amount correlate with condition severity.  
- Predictions remained **fair and consistent** across demographic groups.

---

## 🧠 **Interpretability and Insights**

### **Top Influential Features**
1. **Test Results** — Primary diagnostic indicator  
2. **Admission Type** — Predicts urgency and case severity  
3. **Age** — Older patients tend toward chronic or critical conditions  
4. **Billing Amount** — Reflects complexity and duration of care  
5. **Insurance Provider** — Administrative and procedural patterns  

### **Partial Dependence Findings**
- Higher **Age** and **Billing Amount** increase predicted likelihood of severe medical conditions.  
- Specific **Admission Types** significantly influence the target outcome.

These findings make the model **clinically interpretable** and **actionable**.

---

## ⚖️ **Fairness and Ethics**

- Model tested for **group fairness** across **Gender** and **Blood Type**.  
- Accuracy differences ≤ 5% → ✅ **No demographic bias detected**.  
- Dataset anonymized; identifiers removed → ✅ **Privacy-compliant**.

**Conclusion:**  
The system is **ethically sound**, **trustworthy**, and ready for real-world integration as a clinical decision-support tool.

---

## 🩻 **Key Technical Insights Linked to Healthcare Decisions**

| Technical Insight | Healthcare Application |
|--------------------|-------------------------|
| Correlation between Billing Amount and Room Category | Optimize cost management and resource use |
| Strong link between Admission Type and Condition | Improve triage and patient prioritization |
| Feature independence confirmed | Reliable model generalization across cases |
| Fair and interpretable model | Supports transparent clinical decision-making |
| High predictive performance | Enables early diagnosis support and hospital planning |

---

## 🚀 **How to Run the Project**

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/healthcare-diagnosis-prediction.git
   cd healthcare-diagnosis-prediction


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
# 🎬 Movie Recommendation System | Uneeq Internship Task 3

## 🚀 Project Overview

In this project, I built an **end-to-end movie recommendation system** that learns user preferences and delivers personalised movie suggestions — the same way platforms like Netflix, Amazon Prime, and Disney+ engage their users.

The goal was to explore **collaborative filtering** and **content-based filtering** techniques, evaluate their performance using relevant metrics, and translate these insights into real-world business impact for a streaming platform.

---

## 🎯 Business Problem

Streaming and digital platforms succeed when users consistently find content they love.  
A poor recommendation strategy leads to:
- Low engagement
- Reduced watch time
- Increased churn (subscription cancellations)

### Business Objective
To build a system that can intelligently recommend movies that each user is most likely to enjoy next — **increasing engagement, watch time, and user retention**.

---

## 🧠 Project Goals

| Objective | Description |
|------------|--------------|
| 🎯 Build | Develop a movie recommendation engine using real-world ratings data |
| 🔍 Explore | Apply both **content-based** and **collaborative filtering** methods |
| ⚙️ Evaluate | Measure model accuracy and relevance with **RMSE**, **Precision@K**, and **Recall@K** |
| 💡 Explain | Translate technical findings into business insights and product decisions |
| 🌐 Showcase | Host code on GitHub, record a video walkthrough, and share results on LinkedIn |

---

## 📊 Dataset

- **Source:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Files Used:**
  - `movies.csv` — Contains movie IDs, titles, and genres  
  - `ratings.csv` — Contains user IDs, movie IDs, and ratings

| File | Rows | Columns | Description |
|------|-------|----------|-------------|
| `movies.csv` | ~9,000 | 3 | Movie metadata (title, genres) |
| `ratings.csv` | ~100,000 | 4 | User-item interactions and ratings |

---

## 🧱 Methodology & Techniques

The recommendation pipeline is built using three approaches that mirror real-world production systems.

### 1️⃣ Popularity-Based Model (Baseline)
> “Top 10 most watched by everyone”

- **Logic:** Rank movies by total number of ratings (and break ties using average rating)
- **Use case:** Works well for **new users** (no history yet)
- **Business benefit:** Perfect for homepage banners like “🔥 Trending Now”

---

### 2️⃣ Content-Based Filtering
> “Because you watched Inception…”

- **Logic:**  
  - Represent each movie’s genres using TF-IDF vectorisation  
  - Compute **cosine similarity** between movies  
  - Recommend the most similar titles

- **Tech stack:**  
  `scikit-learn` → `TfidfVectorizer`, `cosine_similarity`

- **Use case:**  
  Works even for **new movies** (cold start for items)

- **Example:**  
  If a user watched *Toy Story*, the model recommends similar animation/family films.

---

### 3️⃣ Collaborative Filtering (Matrix Factorisation using TruncatedSVD)
> “Top picks just for you”

- **Logic:**  
  - Build a user–movie ratings matrix  
  - Use `TruncatedSVD` to learn latent “taste” features  
  - Predict missing ratings  
  - Recommend top N movies with the highest predicted ratings

- **Why TruncatedSVD instead of `surprise` library?**  
  Because `scikit-surprise` has compatibility issues with NumPy 2.0+ in modern Colab.  
  TruncatedSVD provides the same mathematical foundation while being compatible, faster, and clean.

- **Tech stack:**  
  `scikit-learn` → `TruncatedSVD`, `mean_squared_error`

- **Use case:**  
  Personalised feed for returning users.

---

## 🧮 Evaluation Metrics

| Metric | Description | Interpretation |
|---------|--------------|----------------|
| **RMSE** | Root Mean Square Error between actual and predicted ratings | Measures overall accuracy of rating prediction |
| **Precision@10** | Of the top 10 recommended movies, how many were relevant (rating ≥ 4)? | Measures **quality** of recommendations |
| **Recall@10** | Of all movies a user actually liked, how many appeared in the top 10? | Measures **coverage** of relevant items |

### Example Output:
Approximate RMSE (TruncatedSVD CF): 0.88
Mean Precision@10: 0.62
Mean Recall@10: 0.57

✅ **Interpretation:**
- RMSE < 1 → our rating predictions are close to real user opinions.  
- Precision@10 = 0.62 → 6 out of 10 recommended movies are genuinely liked by users.  
- Recall@10 = 0.57 → we’re capturing over half of each user’s favourites.

---

## 📈 Results Summary

| Model | Personalisation | Strength | Limitation |
|--------|------------------|-----------|-------------|
| Popularity-Based | ❌ No | Reliable for new users | Not personalised |
| Content-Based | ✅ Partial | Great for “Because you watched” suggestions | Doesn’t capture cross-genre interests |
| Collaborative Filtering | ✅✅ Full | Learns complex taste patterns | Needs enough user data |

**Best performer:** Collaborative Filtering  
**Best hybrid strategy:** Combine Collaborative + Content-Based → “Hybrid Recommendation System”

---

## 💼 Business Insights

### How this improves platform KPIs:

| Business Metric | Impact |
|------------------|--------|
| **Engagement** | Users find more content they like faster |
| **Retention** | Personalised suggestions reduce churn |
| **Cross-sell** | Enables marketing of underexposed “hidden gems” |
| **New content discovery** | Promotes new releases intelligently |
| **Brand perception** | Users feel “the platform really understands me” |

---

## 🧭 Next Steps / Roadmap

1. **Hybrid Model**
   - Blend collaborative (taste) + content (metadata) approaches for better cold-start handling.

2. **Freshness Boost**
   - Prioritise new releases or trending titles using time-based weighting.

3. **Diversity Enhancement**
   - Introduce penalties for recommending too many similar titles (increase variety).

4. **Continuous Learning**
   - Incorporate real-time watch and click data to retrain models dynamically.

5. **Deployment & A/B Testing**
   - Integrate into a web or app interface and test with real user groups.

---

## ⚙️ Tools & Technologies

| Category | Tools |
|-----------|--------|
| **Languages** | Python |
| **Libraries** | pandas, numpy, scikit-learn, seaborn, matplotlib |
| **Environment** | Google Colab |
| **Version Control** | Git + GitHub |
| **Video Demo** | YouTube (linked in submission) |
| **Showcase** | LinkedIn (tagged @Uneeq Interns) |

---

## 📂 Repository Structure

├── README.md
├── Uneeq_Task3_Recommendation_System.ipynb
├── data/
│ ├── movies.csv
│ ├── ratings.csv
└── outputs/
├── evaluation_metrics.txt
└── visualisations/

---

## 🧾 Submission Checklist

| Requirement | Status |
|--------------|--------|
| ✅ Public GitHub Repository | ✔️ |
| ✅ Notebook with EDA, Models & Results | ✔️ |
| ✅ Business Summary & Evaluation Metrics | ✔️ |
| ✅ YouTube Demo Video | 🔜 |
| ✅ LinkedIn Post Tagging Uneeq Interns | 🔜 |

---

## 💬 Key Takeaway

> “This project bridges data science and business — transforming raw ratings data into meaningful personal experiences that increase engagement, retention, and customer loyalty.”

It demonstrates the power of recommendation systems not just as machine learning models, but as **strategic engines of user satisfaction and platform growth**.


## 👩‍💻 Author  

**Maryam Mohamed**  
Uneeq Internship (ML Track)  


---


