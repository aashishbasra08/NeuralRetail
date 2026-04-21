import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


def run_shap_analysis(X_test, model_path=r"models/xgb_churn.pkl"):

    print("SHAP analysis start...")

    # Ensure reports folder exists
    os.makedirs("reports", exist_ok=True)

    # ===============================
    # Load Model
    # ===============================
    model = joblib.load(model_path)

    # ===============================
    # SHAP Explainer
    # ===============================
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_test)

    # ===============================
    # Plot 1: Feature Importance
    # ===============================
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    plt.title("SHAP Feature Importance - Churn Model")
    plt.tight_layout()
    plt.savefig("reports/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: reports/shap_importance.png")

    # ===============================
    # Plot 2: Beeswarm
    # ===============================
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)

    plt.title("SHAP Beeswarm - Feature Impact Distribution")
    plt.tight_layout()
    plt.savefig("reports/shap_beeswarm.png", dpi=150)
    plt.close()

    print("Saved: reports/shap_beeswarm.png")

    # ===============================
    # Plot 3: Waterfall (Single Customer)
    # ===============================
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

    # ===============================
    # Business Insights
    # ===============================
    feature_names =X_test.columns.tolist()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    print("\n BUSINESS INSIGHTS FROM SHAP")
    print("--------------------------------")

    print(f"Top churn driver: {importance_df.iloc[0]['feature']}")

    print("\n Rules:")
    print("- If recency > 90 → HIGH churn risk")
    print("- If frequency < 3 → MODERATE churn risk")
    print("- If monetary < 100 → LOW retention value")

    print("\n Recommendations:")
    print("- Win-back campaign (recency > 60)")
    print("- Loyalty discount (frequency < 3)")
    print("- VIP treatment (monetary > 500)")

    return shap_values, importance_df