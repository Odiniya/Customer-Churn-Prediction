# Project Title: Customer Churn Prediction

## Project Overview

**Context:**
A fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3.

**Description:**
This project aims to predict customer churn for a telecommunications company. By analyzing customer data, we build a model to identify which customers are likely to leave the company, allowing the company to take proactive measures.

**Objective:**
To develop a predictive model that accurately forecasts customer churn based on various features, such as customer demographics and usage patterns.

## Getting Started

Follow these steps to set up and run the project:

### Installation

1. **Clone the Repository:** Begin by cloning this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/Odiniya/Customer-Churn-Prediction.git
   ```

2. **Navigate to the Directory:** Move into the project directory

   ```bash
   cd Customer-Churn-Prediction
   ```

3. **Create a Virtual Environment (Optional):** While not mandatory, it's recommended to create a virtual environment to isolate the project dependencies. You can create a virtual environment using venv or conda. For venv, execute:

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - **On Windows**:

      ```bash
      venv\Scripts\activate
      ```

   - **On MacOS and Linux**:

      ```bash
      source venv/bin/activate
      ```

4. **Install Dependencies:**

 ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

5. **Run the Jupyter notebook:** Launch Jupyter Notebook to explore the provided dataset and start your analysis:

    ```bash
    jupyter notebook
   ```

6. **Open the Notebook**:

In your web browser, navigate to `http://localhost:8888` to access the Jupyter Notebook interface. Open the following notebooks to begin your work:

- **Exploratory Data Analysis.ipynb**: Gain insights into the dataset.

- **Data Preprocessing-and-Modeling.ipynb**: Begin data transformation and model building.

## Data Exploration

### Telco Customer Churn Dataset

#### Context

This dataset originates from a fictional telecommunications company that provided home phone and Internet services to 7,043 customers in California during the third quarter (Q3). It encompasses various features related to customer demographics, service usage, and billing information. The main objective is to predict customer churn, i.e., whether a customer will leave the company based on the provided features.

#### Dataset Description

The dataset contains information about customer behavior and subscription services, including demographic details, service usage, and billing information. Key columns include `CustomerID`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Tenure Months`, and various service-related features such as `Phone Service`, `Internet Service`, `Online Security`, and `Monthly Charge`.

- **CustomerID**: A unique ID for each customer.
- **Gender**: The customer's gender (Male, Female).
- **Senior Citizen**: Indicates if the customer is 65 or older (Yes, No).
- **Partner**: Indicates if the customer has a partner (Yes, No).
- **Dependents**: Indicates if the customer lives with dependents (Yes, No).
- **Tenure Months**: The total number of months the customer has been with the company.
- **Phone Service**: Indicates if the customer subscribes to home phone service (Yes, No).
- **Multiple Lines**: Indicates if the customer subscribes to multiple phone lines (Yes, No).
- **Internet Service**: Indicates if the customer subscribes to Internet service (No, DSL, Fiber Optic, Cable).
- **Online Security**: Indicates if the customer subscribes to online security (Yes, No).
- **Online Backup**: Indicates if the customer subscribes to online backup (Yes, No).
- **Device Protection**: Indicates if the customer subscribes to device protection (Yes, No).
- **Tech Support**: Indicates if the customer subscribes to tech support (Yes, No).
- **Streaming TV**: Indicates if the customer streams TV from a third party (Yes, No).
- **Streaming Movies**: Indicates if the customer streams movies from a third party (Yes, No).
- **Contract**: The customer's contract type (Month-to-Month, One Year, Two Year).
- **Paperless Billing**: Indicates if the customer uses paperless billing (Yes, No).
- **Payment Method**: How the customer pays their bill (Bank Withdrawal, Credit Card, Mailed Check).
- **Monthly Charge**: The total monthly charge for all services.
- **Total Charges**: The total charges calculated up to the end of the quarter.
- **Churn Label**: Indicates if the customer left the company this quarter (Yes, No).
- **Churn Value**: Binary indicator for churn (1 = churned, 0 = not churned).
- **Churn Score**: A score from 0-100 predicting the likelihood of churn.
- **CLTV**: Customer Lifetime Value, predicting the total value of the customer.
- **Churn Reason**: The specific reason given for leaving the company.

**Data Cleaning:**

- Handled missing values and non-numeric entries.
- Encoded categorical variables and transformed columns as needed.

**Exploratory Data Analysis (EDA):**

- Visualizations and insights into customer demographics, service usage, and churn patterns.

## Feature Engineering

**Feature Creation:**

- Created interaction features such as `MonthlyCharges * TenureMonths` to capture customer behavior.
- Performed clustering to group cities based on geographical coordinates.

**Feature Importance:**

- Analyzed feature importance to understand which variables most influence churn predictions.

## Modeling

**Model Selection:**

- **Algorithm Used:** RandomForestClassifier
- **Rationale:** Chosen for its robustness and ability to handle complex interactions between features.

**Hyperparameter Tuning:**

- Tuned hyperparameters using GridSearchCV to optimize model performance. After tuning Hyperparameters, the accuracy with the best parameters remains the same at `0.93`, suggesting that the model was already performing optimally, and fine-tuning the parameters did not lead to a significant change in overall accuracy.

## Evaluation

**Performance Metrics:**

- **Accuracy:** 0.93
- **Precision, Recall, F1-score:** Detailed in the classification report.
- **Cross-Validation Scores:** [0.925, 0.933, 0.930, 0.933, 0.925]

**Error Analysis:**

- Identified and analyzed misclassified instances to understand model limitations and improve performance.

## Results and Interpretation

**Key Findings:**

- The model demonstrates strong performance in predicting customer churn.
- Key features influencing churn include `Churn Score`, `Tenure Months`, `Total Payment`, `Total Charges`, and `Monthly Charges`.

**Feature Importance:**

- Most influential features have been highlighted and discussed in the context of their impact on churn predictions.

## Conclusion

**Summary:**

- The project successfully developed a predictive model for customer churn with a high level of accuracy.
- Insights from the model can help the company implement targeted retention strategies.

**Future Work:**

- Further improvements could include additional feature engineering, experimenting with other models, and incorporating more recent data.

## Repository Structure

- **`data/`**: Raw and processed datasets.
- **`notebooks`**: Jupyter notebooks with EDA, feature engineering, and modeling.
- **`README.md`**: Instructions on how to run the code and reproduce the results.

## Links and References

- **Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Libraries Used:** scikit-learn, pandas, numpy
