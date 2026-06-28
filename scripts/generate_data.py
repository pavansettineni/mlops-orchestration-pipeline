"""
Generate a synthetic customer churn dataset and save to data/churn.csv.
Run once before training: python scripts/generate_data.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2000


def main():
    monthly_spend = RNG.uniform(10, 500, N)
    tenure_months = RNG.integers(1, 72, N)
    support_tickets = RNG.integers(0, 20, N)
    login_frequency = RNG.integers(0, 30, N)

    # Churn probability rises with support tickets, falls with tenure/logins
    # Intercept tuned to produce ~25% churn for a more balanced dataset
    logit = (
        -0.5
        + 0.5 * (support_tickets / 5)
        - 0.03 * tenure_months
        - 0.06 * login_frequency
        + 0.001 * monthly_spend
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churned = RNG.binomial(1, prob_churn)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(N)],
            "monthly_spend": monthly_spend.round(2),
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "login_frequency": login_frequency,
            "churned": churned,
        }
    )
    df.to_csv("data/churn.csv", index=False)
    print(f"Saved {N} rows → data/churn.csv  (churn rate: {churned.mean():.1%})")


if __name__ == "__main__":
    main()
