# src/models/shap_analysis.py
# This file runs SHAP explainability analysis on the trained churn (or demand) model.
# It generates three visual reports — feature importance bar chart, beeswarm plot,
# and a waterfall explanation for a single customer — and prints business rules
# derived from the top SHAP features.

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


def run_shap_analysis(X_test, model_path=r"models/xgb_churn.pkl"):

    print("SHAP analysis start...")

    # Create reports folder if it does not already exist
    os.makedirs("reports", exist_ok=True)

    # Load the trained model from disk
    model = joblib.load(model_path)

    # Initialize TreeExplainer — optimized for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_test)

    # Plot 1: Bar chart showing mean absolute SHAP value per feature (global importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - Churn Model")
    plt.tight_layout()
    plt.savefig("reports/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_importance.png")

    # Plot 2: Beeswarm plot showing how each feature value impacts predictions across all samples
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title("SHAP Beeswarm - Feature Impact Distribution")
    plt.tight_layout()
    plt.savefig("reports/shap_beeswarm.png", dpi=150)
    plt.close()
    print("Saved: reports/shap_beeswarm.png")

    # Plot 3: Waterfall plot for a single customer — explains why that prediction was made
    customer_idx = 0
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[customer_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[customer_idx],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f"Customer {customer_idx} - Churn Explanation")
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_customer0.png", dpi=150)
    plt.close()
    print("Saved: reports/shap_waterfall_customer0.png")

    # Compute mean absolute SHAP value per feature to rank overall importance
    feature_names = X_test.columns.tolist()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    print("\n BUSINESS INSIGHTS FROM SHAP")
    print("--------------------------------")
    print(f"Top churn driver: {importance_df.iloc[0]['feature']}")

    top_feature = importance_df.iloc[0]['feature']
    print("\n Rules:")

    # Detect whether this is a demand model or churn model based on top feature name
    if 'demand_diff' in top_feature or 'lag' in top_feature or 'rolling' in top_feature:
        # Rules for demand forecasting model
        print("- If demand_diff < 0 → Demand DECLINING trend")
        print("- If rolling_mean_7 < rolling_mean_3 → SHORT-TERM spike only")
        print("- If lag_1 high → Previous day demand strong")
        print("\n Recommendations:")
        print("- Restock alert (demand_diff < -50)")
        print("- Promo campaign (rolling_mean_7 declining)")
        print("- Maintain inventory (lag_1 > avg demand)")
    else:
        # Rules for churn prediction model
        print("- If recency > 90 → HIGH churn risk")
        print("- If frequency < 3 → MODERATE churn risk")
        print("- If monetary < 100 → LOW retention value")
        print("\n Recommendations:")
        print("- Win-back campaign (recency > 60)")
        print("- Loyalty discount (frequency < 3)")
        print("- VIP treatment (monetary > 500)")

        return shap_values, importance_df


if __name__ == "__main__":
    print("Running SHAP Analysis Script...")

    import os

    # Check that the test data file exists before proceeding
    if not os.path.exists("data/X_test.csv"):
        print("ERROR: data/X_test.csv not found")
        print("Run churn.py first to generate this file")
        exit()

    # Load test features and run the full SHAP analysis pipeline
    X_test = pd.read_csv("data/X_test.csv")
    run_shap_analysis(X_test)