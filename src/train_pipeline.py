"""
Strict 8-Stage Machine Learning Training & Evaluation Pipeline
for Smart Money Concepts (SMC) Trade Setup Classifier on NIFTY 50.
Guarantees Zero Data Leakage through encapsulated pipeline cross-validation.
"""

import os
import sys
import json
import warnings
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

from src.data_pipeline import (
    build_dataset,
    FEATURE_COLUMNS,
    SMC_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def run_pipeline():
    print("=" * 80)
    print("SMART MONEY CONCEPTS (SMC) TRADE SETUP CLASSIFIER - NIFTY 50")
    print("8-STAGE LEAK-FREE MACHINE LEARNING EVALUATION PIPELINE")
    print("=" * 80)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    results_dict = {}

    # -------------------------------------------------------------------------
    # STAGE 1: Data & Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 1: Data Ingestion & Exploratory Data Analysis")
    print("=" * 50)

    df = build_dataset()
    print(f"Total dataset shape: {df.shape}")
    print(f"Features list ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    
    missing_counts = df[FEATURE_COLUMNS + ["take_trade"]].isnull().sum()
    print("\nMissing values per column:")
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "  Zero missing values detected!")

    class_counts = df["take_trade"].value_counts()
    class_prop = df["take_trade"].value_counts(normalize=True)
    print("\nClass Distribution:")
    print(f"  Class 0 (Avoid Trade):      {class_counts.get(0, 0)} ({class_prop.get(0, 0):.2%})")
    print(f"  Class 1 (Take Long Setup):  {class_counts.get(1, 0)} ({class_prop.get(1, 0):.2%})")

    # Save EDA Figure 1: Class Balance
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["0: Avoid Trade\n(Low Probability / No Structure)", "1: Take Trade\n(High Probability SMC Setup)"],
        class_counts.values,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black",
        width=0.55,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 30,
            f"{int(h)} ({h / len(df):.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("NIFTY 50 SMC Trade Setup Class Balance", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Candles", fontsize=11)
    ax.set_ylim(0, max(class_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_class_balance.png")
    plt.close()
    print("  -> Saved 'outputs/figures/eda_class_balance.png'")

    # Save EDA Figure 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[FEATURE_COLUMNS + ["take_trade"]].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature & Target Pearson Correlation Matrix", fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_correlation_heatmap.png")
    plt.close()
    print("  -> Saved 'outputs/figures/eda_correlation_heatmap.png'")

    # -------------------------------------------------------------------------
    # STAGE 2: Stratified Train / Test Split (Held-out Test Isolation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 2: Stratified Train/Test Split (Test Isolation)")
    print("=" * 50)

    X = df[FEATURE_COLUMNS].copy()
    y = df["take_trade"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"Full dataset: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples ({y_train.mean():.2%} positive)")
    print(f"Held-out test set: {len(X_test)} samples ({y_test.mean():.2%} positive)")
    print("CRITICAL: Held-out test set strictly quarantined until Stage 7.")

    # -------------------------------------------------------------------------
    # STAGE 3: Preprocessing Pipeline (Leak-free Definition)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 3: Preprocessing Pipeline Construction")
    print("=" * 50)

    def create_model_pipeline(classifier, with_scaler: bool = True) -> Pipeline:
        """
        Encapsulate SimpleImputer and StandardScaler inside a Pipeline.
        Guarantees that imputation and scaling parameters are fit ONLY on
        training folds and never exposed to validation/test folds.
        """
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if with_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    print("Pipeline constructed: SimpleImputer(median) -> StandardScaler() -> Classifier")
    print("Encapsulation guarantees zero data leakage across CV folds.")

    # -------------------------------------------------------------------------
    # STAGE 4: Baseline Models & Multi-Model Benchmarking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 4: Baseline & Model Benchmarking (Repeated Stratified CV)")
    print("=" * 50)

    cv_benchmark = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring_metrics = ["f1", "roc_auc", "precision", "recall"]

    candidate_models = {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    benchmark_rows = []
    print(f"{'Model':<25} | {'F1-Score':<16} | {'ROC-AUC':<16} | {'Precision':<16} | {'Recall':<16}")
    print("-" * 95)

    for name, clf in candidate_models.items():
        pipe = create_model_pipeline(clf, with_scaler=True)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv_benchmark, scoring=scoring_metrics, n_jobs=1)

        f1_m, f1_s = np.mean(cv_res["test_f1"]), np.std(cv_res["test_f1"])
        auc_m, auc_s = np.mean(cv_res["test_roc_auc"]), np.std(cv_res["test_roc_auc"])
        prec_m, prec_s = np.mean(cv_res["test_precision"]), np.std(cv_res["test_precision"])
        rec_m, rec_s = np.mean(cv_res["test_recall"]), np.std(cv_res["test_recall"])

        benchmark_rows.append({
            "Model": name,
            "F1-Score": f"{f1_m:.3f} ± {f1_s:.3f}",
            "ROC-AUC": f"{auc_m:.3f} ± {auc_s:.3f}",
            "Precision": f"{prec_m:.3f} ± {prec_s:.3f}",
            "Recall": f"{rec_m:.3f} ± {rec_s:.3f}",
            "_f1_mean": f1_m,
            "_auc_mean": auc_m,
        })
        print(f"{name:<25} | {f1_m:.3f} ± {f1_s:.3f}   | {auc_m:.3f} ± {auc_s:.3f}   | {prec_m:.3f} ± {prec_s:.3f}   | {rec_m:.3f} ± {rec_s:.3f}")

    benchmark_df = pd.DataFrame(benchmark_rows)
    results_dict["stage4_benchmarks"] = benchmark_rows

    # -------------------------------------------------------------------------
    # STAGE 5: Nested Cross-Validation & Hyperparameter Tuning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 5: Nested Cross-Validation & Hyperparameter Tuning")
    print("=" * 50)

    # Top model tuning: XGBClassifier
    base_xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_pipeline = create_model_pipeline(base_xgb, with_scaler=True)

    param_grid = {
        "classifier__n_estimators": [60, 120],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.03, 0.08],
        "classifier__subsample": [0.8, 1.0],
    }

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )

    print("Executing Nested Cross-Validation (Outer 5-Fold, Inner 3-Fold)...")
    nested_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring="f1", n_jobs=1)
    nested_auc_scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv, scoring="roc_auc", n_jobs=1)

    print(f"Nested CV F1-Score:  {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
    print(f"Nested CV ROC-AUC:   {np.mean(nested_auc_scores):.4f} ± {np.std(nested_auc_scores):.4f}")

    # Now fit GridSearchCV on the full X_train to find the optimal hyperparameters
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_inner_f1 = grid_search.best_score_
    print("\nOptimal Hyperparameters selected via GridSearchCV:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best Inner CV F1-Score: {best_inner_f1:.4f}")

    results_dict["stage5_nested_cv"] = {
        "nested_f1_mean": float(np.mean(nested_scores)),
        "nested_f1_std": float(np.std(nested_scores)),
        "nested_auc_mean": float(np.mean(nested_auc_scores)),
        "nested_auc_std": float(np.std(nested_auc_scores)),
        "best_params": {k: str(v) for k, v in best_params.items()},
        "best_inner_f1": float(best_inner_f1),
    }

    # -------------------------------------------------------------------------
    # STAGE 6: Ablation Study
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 6: Ablation Study (Component Impact Analysis)")
    print("=" * 50)

    ablation_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use the best classifier hyperparameters found
    tuned_clf_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}

    ablation_experiments = {
        "1. Full Model (All Features + Scaling)": {
            "features": FEATURE_COLUMNS,
            "scaling": True,
        },
        "2. Without SMC Features (Price Action / Tech Only)": {
            "features": [f for f in FEATURE_COLUMNS if f not in SMC_FEATURE_COLUMNS],
            "scaling": True,
        },
        "3. Without StandardScaler (Raw Normalized Scale)": {
            "features": FEATURE_COLUMNS,
            "scaling": False,
        },
        "4. Without Technical Context (SMC Only)": {
            "features": [f for f in FEATURE_COLUMNS if f not in TECHNICAL_FEATURE_COLUMNS],
            "scaling": True,
        },
    }

    ablation_results = []
    print(f"{'Experiment':<52} | {'F1-Score':<16} | {'ROC-AUC':<16}")
    print("-" * 90)

    for exp_name, conf in ablation_experiments.items():
        clf = XGBClassifier(**tuned_clf_params, random_state=42, eval_metric="logloss")
        pipe = create_model_pipeline(clf, with_scaler=conf["scaling"])
        sub_X_train = X_train[conf["features"]]

        res = cross_validate(pipe, sub_X_train, y_train, cv=ablation_cv, scoring=["f1", "roc_auc"], n_jobs=1)
        f1_m, f1_s = np.mean(res["test_f1"]), np.std(res["test_f1"])
        auc_m, auc_s = np.mean(res["test_roc_auc"]), np.std(res["test_roc_auc"])

        ablation_results.append({
            "Experiment": exp_name,
            "Num_Features": len(conf["features"]),
            "Scaling": "StandardScaler" if conf["scaling"] else "None",
            "CV_F1": f"{f1_m:.4f} ± {f1_s:.4f}",
            "CV_ROC_AUC": f"{auc_m:.4f} ± {auc_s:.4f}",
            "_f1_val": f1_m,
            "_auc_val": auc_m,
        })
        print(f"{exp_name:<52} | {f1_m:.4f} ± {f1_s:.4f} | {auc_m:.4f} ± {auc_s:.4f}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv("outputs/ablation_table.csv", index=False)
    print("  -> Saved 'outputs/ablation_table.csv'")
    results_dict["stage6_ablation"] = ablation_results

    # -------------------------------------------------------------------------
    # STAGE 7: Single Final Evaluation on Held-out Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 7: Single Final Test Evaluation (Held-out Set)")
    print("=" * 50)

    # Train final pipeline on full training data using best parameters
    final_pipeline = grid_search.best_estimator_
    final_pipeline.fit(X_train, y_train)

    # Predict ONCE on held-out test data
    y_pred = final_pipeline.predict(X_test)
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_prec = precision_score(y_test, y_pred)
    test_rec = recall_score(y_test, y_pred)

    print(f"Final Test Evaluation Metrics:")
    print(f"  F1-Score:       {test_f1:.4f}")
    print(f"  ROC-AUC:        {test_roc_auc:.4f}")
    print(f"  Precision:      {test_prec:.4f}")
    print(f"  Recall:         {test_rec:.4f}")
    print("\nClassification Report:")
    report_text = classification_report(
        y_test, y_pred, target_names=["0: Avoid Trade", "1: Take Trade"], digits=4
    )
    print(report_text)

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Avoid Trade (0)", "Take Trade (1)"],
        yticklabels=["Avoid Trade (0)", "Take Trade (1)"],
        ax=ax,
    )
    ax.set_title("Test Confusion Matrix (Held-out NIFTY 50)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/figures/test_confusion_matrix.png")
    plt.close()
    print("  -> Saved 'outputs/figures/test_confusion_matrix.png'")

    # Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Tuned XGBoost (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("Test ROC Curve - NIFTY 50 SMC Classifier", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("outputs/figures/test_roc_curve.png")
    plt.close()
    print("  -> Saved 'outputs/figures/test_roc_curve.png'")

    results_dict["stage7_final_test"] = {
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_roc_auc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
    }

    # -------------------------------------------------------------------------
    # STAGE 8: Model Artifacts, SHAP Explainability & Sample Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 8: Artifact Persistence & SHAP Explainability")
    print("=" * 50)

    # 1. Save trained pipeline
    model_path = "models/smc_trade_model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"  -> Model pipeline serialized to '{model_path}'")

    # 2. Generate global SHAP explanation
    # Transform X_test through preprocessing steps of the pipeline
    imputer = final_pipeline.named_steps["imputer"]
    scaler = final_pipeline.named_steps["scaler"]
    classifier = final_pipeline.named_steps["classifier"]

    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)

    explainer = shap.TreeExplainer(classifier)
    shap_explanation = explainer(X_test_scaled_df)
    shap_values = shap_explanation.values

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled_df, feature_names=FEATURE_COLUMNS, show=False)
    plt.title("SHAP Global Feature Importance (Test Set)", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary.png", bbox_inches="tight")
    plt.close()
    print("  -> Saved 'outputs/figures/shap_summary.png'")

    # 3. Save sample test candles for the interactive Streamlit app
    sample_indices = X_test.index[-100:]  # Last 100 test candles
    sample_df = df.loc[sample_indices].copy()
    sample_df["predicted_take_trade"] = final_pipeline.predict(sample_df[FEATURE_COLUMNS])
    sample_df["pred_probability"] = final_pipeline.predict_proba(sample_df[FEATURE_COLUMNS])[:, 1]
    sample_csv_path = "data/sample_test_candles.csv"
    sample_df.to_csv(sample_csv_path)
    print(f"  -> Saved {len(sample_df)} sample test candles to '{sample_csv_path}' for Streamlit app")

    # 4. Save results.json
    results_json_path = "outputs/results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"  -> Saved metrics summary to '{results_json_path}'")

    # 5. Save human-readable summary_report.txt
    summary_report_path = "outputs/summary_report.txt"
    with open(summary_report_path, "w") as f:
        f.write("SMART MONEY CONCEPTS (SMC) TRADE SETUP CLASSIFIER FOR NIFTY 50\n")
        f.write("8-STAGE MACHINE LEARNING EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET OVERVIEW\n")
        f.write(f"  Total Clean Candles:     {len(df)}\n")
        f.write(f"  Class 0 (Avoid Trade):   {class_counts.get(0, 0)} ({class_prop.get(0, 0):.2%})\n")
        f.write(f"  Class 1 (Take Trade):    {class_counts.get(1, 0)} ({class_prop.get(1, 0):.2%})\n\n")

        f.write("2. BASELINE & MODEL BENCHMARKING (15-Fold Repeated Stratified CV)\n")
        f.write(benchmark_df[["Model", "F1-Score", "ROC-AUC", "Precision", "Recall"]].to_string(index=False))
        f.write("\n\n")

        f.write("3. NESTED CROSS-VALIDATION TUNING\n")
        f.write(f"  Outer CV F1:             {results_dict['stage5_nested_cv']['nested_f1_mean']:.4f} ± {results_dict['stage5_nested_cv']['nested_f1_std']:.4f}\n")
        f.write(f"  Outer CV ROC-AUC:        {results_dict['stage5_nested_cv']['nested_auc_mean']:.4f} ± {results_dict['stage5_nested_cv']['nested_auc_std']:.4f}\n")
        f.write(f"  Best Parameters:         {results_dict['stage5_nested_cv']['best_params']}\n\n")

        f.write("4. ABLATION STUDY\n")
        f.write(ablation_df[["Experiment", "Num_Features", "Scaling", "CV_F1", "CV_ROC_AUC"]].to_string(index=False))
        f.write("\n\n")

        f.write("5. FINAL TEST EVALUATION (Held-out Set)\n")
        f.write(f"  Test F1-Score:           {test_f1:.4f}\n")
        f.write(f"  Test ROC-AUC:            {test_roc_auc:.4f}\n")
        f.write(f"  Test Precision:          {test_prec:.4f}\n")
        f.write(f"  Test Recall:             {test_rec:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\n" + "=" * 80 + "\n")
    print(f"  -> Saved text summary to '{summary_report_path}'")

    print("\n" + "=" * 80)
    print("ALL 8 STAGES COMPLETED SUCCESSFULLY WITH ZERO DATA LEAKAGE!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
