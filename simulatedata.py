import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)  # for reproducibility


def generate_credit_risk_data(n_samples=1000):
    # Generate base features with realistic distributions

    age = np.random.randint(18, 70, size=n_samples)  # Age between 18 and 70
    income = np.random.normal(loc=60000, scale=15000, size=n_samples).clip(10000, None)  # Annual income
    loan_amount = np.random.normal(loc=15000, scale=7000, size=n_samples).clip(1000, None)  # Loan amount
    loan_term = np.random.choice([12, 24, 36, 48, 60], size=n_samples)  # Loan term in months
    credit_history_length = np.random.randint(1, 30, size=n_samples)  # Years of credit history
    number_of_defaults = np.random.poisson(0.2, size=n_samples).clip(0, 5)  # Number of past defaults
    debt_to_income_ratio = (loan_amount / income).clip(0, 1)  # Ratio of loan to income
    employment_status = np.random.choice(['employed', 'unemployed', 'self-employed'], size=n_samples, p=[0.7, 0.1, 0.2])

    # Encode employment_status to numeric for classification
    employment_status_map = {'employed': 0, 'unemployed': 1, 'self-employed': 2}
    employment_status_num = np.array([employment_status_map[e] for e in employment_status])

    # Combine features into an array for classification target generation
    X_for_target = np.column_stack([
        age, income, loan_amount, loan_term, credit_history_length,
        number_of_defaults, debt_to_income_ratio, employment_status_num
    ])

    # Generate a binary target variable (0 = good credit, 1 = bad credit)
    # Use make_classification with weights to simulate imbalance
    X_dummy, y = make_classification(
        n_samples=n_samples,
        n_features=X_for_target.shape[1],
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.7, 0.3],  # 70% good, 30% bad credit risk
        class_sep=1.5,
        random_state=42
    )

    # To make y correlated with our features, we can tweak by thresholding a weighted sum instead:
    risk_score = (
            0.03 * age
            - 0.00005 * income
            + 0.0001 * loan_amount
            + 0.01 * loan_term
            - 0.02 * credit_history_length
            + 0.4 * number_of_defaults
            + 1.5 * debt_to_income_ratio
            + 0.2 * employment_status_num
            +np.random.normal(0,0.2)
    )
    y = (risk_score > np.percentile(risk_score, 70)).astype(int)  # top 30% risk labeled as bad (1)

    # Assemble DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_history_length': credit_history_length,
        'number_of_defaults': number_of_defaults,
        'debt_to_income_ratio': debt_to_income_ratio,
        'employment_status': employment_status,
        'credit_risk': y
    })

    return df


# Example usage
data = generate_credit_risk_data(1000)
data.to_csv('./data.csv')
