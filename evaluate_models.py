"""
evaluate_models.py
Comprehensive model evaluation and comparison script for all trained models.
Generates comparison reports, visualizations, and insights for paper presentation.

Usage: python evaluate_models.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────

MODEL_DIR = "models/saved"
DATA_DIR = "data"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ─────────────────────────────────────────────────────────────────────────

def load_reports():
    """Load all comparison reports generated during training."""
    reports = {}
    
    try:
        with open(os.path.join(MODEL_DIR, 'demand_comparison_report.json')) as f:
            reports['demand'] = json.load(f)
        logger.info("✅ Loaded demand comparison report")
    except FileNotFoundError:
        logger.warning("⚠️  Demand comparison report not found")
    
    try:
        with open(os.path.join(MODEL_DIR, 'delay_comparison_report.json')) as f:
            reports['delay'] = json.load(f)
        logger.info("✅ Loaded delay comparison report")
    except FileNotFoundError:
        logger.warning("⚠️  Delay comparison report not found")
    
    try:
        with open(os.path.join(MODEL_DIR, 'anomaly_comparison_report.json')) as f:
            reports['anomaly'] = json.load(f)
        logger.info("✅ Loaded anomaly comparison report")
    except FileNotFoundError:
        logger.warning("⚠️  Anomaly comparison report not found")
    
    return reports

def create_demand_comparison_viz(report):
    """Create demand model comparison visualizations."""
    logger.info("\n📊 Creating demand model comparison visualizations...")
    
    models = report['models']
    comparison_data = []
    
    for model_name, metrics in models.items():
        if isinstance(metrics, dict) and 'mae' in metrics:
            comparison_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2'],
            })
    
    if not comparison_data:
        logger.warning("   No demand model data to visualize")
        return
    
    df_comp = pd.DataFrame(comparison_data).sort_values('MAE')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Demand Prediction - Model Comparison', fontsize=16, fontweight='bold')
    
    # MAE
    axes[0, 0].barh(df_comp['Model'], df_comp['MAE'], color='steelblue')
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0, 0].set_xlabel('MAE (passengers)')
    
    # RMSE
    axes[0, 1].barh(df_comp['Model'], df_comp['RMSE'], color='coral')
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    axes[0, 1].set_xlabel('RMSE (passengers)')
    
    # MAPE
    axes[1, 0].barh(df_comp['Model'], df_comp['MAPE'] * 100, color='seagreen')
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)', fontweight='bold')
    axes[1, 0].set_xlabel('MAPE (%)')
    
    # R²
    axes[1, 1].barh(df_comp['Model'], df_comp['R²'], color='mediumpurple')
    axes[1, 1].set_title('R² Score', fontweight='bold')
    axes[1, 1].set_xlabel('R²')
    axes[1, 1].axvline(0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'demand_model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✅ Saved: {plot_path}")
    
    # Export CSV
    csv_path = os.path.join(OUTPUT_DIR, 'demand_comparison.csv')
    df_comp.to_csv(csv_path, index=False)
    logger.info(f"   ✅ Exported: {csv_path}")
    
    return df_comp

def create_delay_comparison_viz(report):
    """Create delay model comparison visualizations."""
    logger.info("\n📊 Creating delay model comparison visualizations...")
    
    all_metrics = report['regression_metrics']
    reg_metrics = {k: v for k, v in all_metrics.items() if 'reg' in k}
    clf_metrics = {k: v for k, v in all_metrics.items() if 'clf' in k}
    
    # Regression comparison
    reg_data = []
    for model_name, metrics in reg_metrics.items():
        if isinstance(metrics, dict):
            reg_data.append({
                'Model': model_name.replace('_reg', ''),
                'MAE (min)': metrics['mae'],
                'RMSE (min)': metrics['rmse'],
                'R²': metrics['r2'],
            })
    
    if reg_data:
        df_reg = pd.DataFrame(reg_data).sort_values('MAE (min)')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Delay Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # Regression metrics
        axes[0, 0].barh(df_reg['Model'], df_reg['MAE (min)'], color='steelblue')
        axes[0, 0].set_title('Regression MAE', fontweight='bold')
        axes[0, 0].set_xlabel('MAE (minutes)')
        
        axes[0, 1].barh(df_reg['Model'], df_reg['RMSE (min)'], color='coral')
        axes[0, 1].set_title('Regression RMSE', fontweight='bold')
        axes[0, 1].set_xlabel('RMSE (minutes)')
        
        axes[1, 0].barh(df_reg['Model'], df_reg['R²'], color='seagreen')
        axes[1, 0].set_title('Regression R² Score', fontweight='bold')
        axes[1, 0].set_xlabel('R²')
        
        # Classification metrics
        clf_data = []
        for model_name, metrics in clf_metrics.items():
            if isinstance(metrics, dict):
                clf_data.append({
                    'Model': model_name.replace('_clf', ''),
                    'F1': metrics['f1'],
                    'AUC': metrics['auc'],
                })
        
        if clf_data:
            df_clf = pd.DataFrame(clf_data).sort_values('F1', ascending=False)
            x = np.arange(len(df_clf))
            width = 0.35
            axes[1, 1].bar(x - width/2, df_clf['F1'], width, label='F1-Score', color='mediumpurple')
            axes[1, 1].bar(x + width/2, df_clf['AUC'], width, label='AUC-ROC', color='darkorange')
            axes[1, 1].set_title('Classification Metrics (Is Delayed)', fontweight='bold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(df_clf['Model'], rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'delay_model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ Saved: {plot_path}")
        
        csv_path = os.path.join(OUTPUT_DIR, 'delay_comparison.csv')
        df_reg.to_csv(csv_path, index=False)
        logger.info(f"   ✅ Exported: {csv_path}")

def create_anomaly_comparison_viz(report):
    """Create anomaly detection model comparison visualizations."""
    logger.info("\n📊 Creating anomaly detection comparison visualizations...")
    
    models = report['models']
    comp_data = []
    
    for model_name, metrics in models.items():
        if isinstance(metrics, dict) and 'f1' in metrics:
            comp_data.append({
                'Model': model_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'Specificity': metrics['specificity'],
            })
    
    if not comp_data:
        logger.warning("   No anomaly model data to visualize")
        return
    
    df_comp = pd.DataFrame(comp_data).sort_values('F1-Score', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection - Model Comparison', fontsize=16, fontweight='bold')
    
    # F1-Score
    axes[0, 0].barh(df_comp['Model'], df_comp['F1-Score'], color='steelblue')
    axes[0, 0].set_title('F1-Score', fontweight='bold')
    axes[0, 0].set_xlabel('F1-Score')
    
    # Precision vs Recall
    x = np.arange(len(df_comp))
    width = 0.35
    axes[0, 1].bar(x - width/2, df_comp['Precision'], width, label='Precision', color='coral')
    axes[0, 1].bar(x + width/2, df_comp['Recall'], width, label='Recall', color='seagreen')
    axes[0, 1].set_title('Precision vs Recall', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_comp['Model'], rotation=45)
    axes[0, 1].legend()
    
    # Specificity
    axes[1, 0].barh(df_comp['Model'], df_comp['Specificity'], color='mediumpurple')
    axes[1, 0].set_title('Specificity (True Negative Rate)', fontweight='bold')
    axes[1, 0].set_xlabel('Specificity')
    
    # All metrics radar
    metrics_cols = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    for i, row in df_comp.head(3).iterrows():
        axes[1, 1].plot(metrics_cols, [row[col] for col in metrics_cols], marker='o', label=row['Model'])
    axes[1, 1].set_title('Top 3 Models - Metrics Profile', fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'anomaly_model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✅ Saved: {plot_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, 'anomaly_comparison.csv')
    df_comp.to_csv(csv_path, index=False)
    logger.info(f"   ✅ Exported: {csv_path}")

def create_summary_report(reports):
    """Create a comprehensive summary report."""
    logger.info("\n📄 Creating summary report...")
    
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "datasets": {},
        "task_summaries": {}
    }
    
    # Load datasets info
    if os.path.exists(os.path.join(DATA_DIR, 'demand_dataset.csv')):
        demand_df = pd.read_csv(os.path.join(DATA_DIR, 'demand_dataset.csv'))
        summary['datasets']['demand'] = {
            "rows": len(demand_df),
            "years": 3,
            "routes": demand_df['route_id'].nunique(),
        }
    
    if os.path.exists(os.path.join(DATA_DIR, 'delay_dataset.csv')):
        delay_df = pd.read_csv(os.path.join(DATA_DIR, 'delay_dataset.csv'))
        summary['datasets']['delay'] = {
            "rows": len(delay_df),
            "years": 3,
            "routes": delay_df['route_id'].nunique(),
        }
    
    if os.path.exists(os.path.join(DATA_DIR, 'anomaly_dataset.csv')):
        anomaly_df = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_dataset.csv'))
        summary['datasets']['anomaly'] = {
            "rows": len(anomaly_df),
            "anomaly_rate": float(anomaly_df['is_anomaly'].mean() * 100),
        }
    
    # Task summaries
    if 'demand' in reports:
        demand_models = reports['demand']['models']
        valid_models = {k: v for k, v in demand_models.items() if isinstance(v, dict) and 'mae' in v}
        if valid_models:
            best = min(valid_models.items(), key=lambda x: x[1]['mae'])
            summary['task_summaries']['demand'] = {
                "best_model": best[0],
                "best_mae": float(best[1]['mae']),
                "best_r2": float(best[1]['r2']),
                "models_compared": len(valid_models),
            }
    
    if 'delay' in reports:
        delay_metrics = reports['delay']['regression_metrics']
        reg_models = {k: v for k, v in delay_metrics.items() if 'reg' in k and isinstance(v, dict)}
        if reg_models:
            best = min(reg_models.items(), key=lambda x: x[1]['mae'])
            summary['task_summaries']['delay'] = {
                "best_model": best[0].replace('_reg', ''),
                "best_mae_minutes": float(best[1]['mae']),
                "models_compared": len(reg_models),
            }
    
    if 'anomaly' in reports:
        anomaly_models = reports['anomaly']['models']
        valid_models = {k: v for k, v in anomaly_models.items() if isinstance(v, dict) and 'f1' in v}
        if valid_models:
            best = max(valid_models.items(), key=lambda x: x[1]['f1'])
            summary['task_summaries']['anomaly'] = {
                "best_model": best[0],
                "best_f1": float(best[1]['f1']),
                "anomaly_rate": float(reports['anomaly'].get('anomaly_rate', 0)),
                "models_compared": len(valid_models),
            }
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"   ✅ Saved: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    if 'demand' in summary['task_summaries']:
        d = summary['task_summaries']['demand']
        print(f"\n🎯 DEMAND PREDICTION")
        print(f"   Best Model: {d['best_model']}")
        print(f"   MAE: {d['best_mae']:.2f} passengers")
        print(f"   R² Score: {d['best_r2']:.4f}")
        print(f"   Models Compared: {d['models_compared']}")
    
    if 'delay' in summary['task_summaries']:
        d = summary['task_summaries']['delay']
        print(f"\n🎯 DELAY PREDICTION")
        print(f"   Best Model: {d['best_model']}")
        print(f"   MAE: {d['best_mae_minutes']:.2f} minutes")
        print(f"   Models Compared: {d['models_compared']}")
    
    if 'anomaly' in summary['task_summaries']:
        d = summary['task_summaries']['anomaly']
        print(f"\n🎯 ANOMALY DETECTION")
        print(f"   Best Model: {d['best_model']}")
        print(f"   F1-Score: {d['best_f1']:.3f}")
        print(f"   Anomaly Detection Rate: {d['anomaly_rate']:.2f}%")
        print(f"   Models Compared: {d['models_compared']}")
    
    print("\n" + "="*80)
    print(f"✅ Evaluation results saved to: {OUTPUT_DIR}/")
    print("="*80)

    return summary

def create_executive_dashboard(summary):
    """Create a single visual dashboard for Colab and paper presentation."""
    logger.info("\n🧭 Creating executive dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("SmartDTC AI Service — Executive Model Dashboard", fontsize=18, fontweight="bold")

    # Demand summary card
    axes[0, 0].axis("off")
    demand = summary.get("task_summaries", {}).get("demand")
    if demand:
        demand_text = (
            f"Demand Prediction\n\n"
            f"Best Model: {demand['best_model'].upper()}\n"
            f"MAE: {demand['best_mae']:.2f} passengers\n"
            f"R²: {demand['best_r2']:.4f}\n"
            f"Models Compared: {demand['models_compared']}"
        )
    else:
        demand_text = "Demand Prediction\n\nNo report available"
    axes[0, 0].text(0.02, 0.95, demand_text, va="top", ha="left", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#EFF6FF", edgecolor="#93C5FD"))

    # Delay summary card
    axes[0, 1].axis("off")
    delay = summary.get("task_summaries", {}).get("delay")
    if delay:
        delay_text = (
            f"Delay Prediction\n\n"
            f"Best Model: {delay['best_model'].upper()}\n"
            f"MAE: {delay['best_mae_minutes']:.2f} minutes\n"
            f"Models Compared: {delay['models_compared']}"
        )
    else:
        delay_text = "Delay Prediction\n\nNo report available"
    axes[0, 1].text(0.02, 0.95, delay_text, va="top", ha="left", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECFDF5", edgecolor="#86EFAC"))

    # Anomaly summary card
    axes[1, 0].axis("off")
    anomaly = summary.get("task_summaries", {}).get("anomaly")
    if anomaly:
        anomaly_text = (
            f"Anomaly Detection\n\n"
            f"Best Model: {anomaly['best_model'].upper()}\n"
            f"F1-Score: {anomaly['best_f1']:.3f}\n"
            f"Anomaly Rate: {anomaly['anomaly_rate']:.2f}%\n"
            f"Models Compared: {anomaly['models_compared']}"
        )
    else:
        anomaly_text = "Anomaly Detection\n\nNo report available"
    axes[1, 0].text(0.02, 0.95, anomaly_text, va="top", ha="left", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF7ED", edgecolor="#FDBA74"))

    # Overall recommendations
    axes[1, 1].axis("off")
    recommendations = [
        "Use the PNG charts for slides",
        "Use the CSV tables for your paper",
        "Use the JSON summary for appendix",
        "Download SmartDTC_Model_Comparison_Results.zip",
    ]
    rec_text = "What to export\n\n" + "\n".join(f"• {item}" for item in recommendations)
    axes[1, 1].text(0.02, 0.95, rec_text, va="top", ha="left", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F3FF", edgecolor="#C4B5FD"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    dashboard_path = os.path.join(OUTPUT_DIR, 'executive_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✅ Saved: {dashboard_path}")

# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("╔════════════════════════════════════════════════════════════════╗")
    logger.info("║   SmartDTC AI Service - Model Evaluation & Comparison         ║")
    logger.info("╚════════════════════════════════════════════════════════════════╝")
    
    # Load reports
    reports = load_reports()
    
    if not reports:
        logger.error("❌ No comparison reports found. Run model training first.")
        exit(1)
    
    # Create visualizations
    if 'demand' in reports:
        create_demand_comparison_viz(reports['demand'])
    
    if 'delay' in reports:
        create_delay_comparison_viz(reports['delay'])
    
    if 'anomaly' in reports:
        create_anomaly_comparison_viz(reports['anomaly'])
    
    # Create summary
    summary = create_summary_report(reports)
    create_executive_dashboard(summary)
    
    logger.info("\n✨ Evaluation complete!")
