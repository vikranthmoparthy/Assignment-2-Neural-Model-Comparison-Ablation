import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_learning_curves(history: list, save_path: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="green", marker="^")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1 Score")
    ax2.set_title("Validation Metric")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    
    plt.savefig(save_path)
    plt.close()

def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)

def extract_and_save_errors(y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame, save_path: str, num_errors: int = 15) -> pd.DataFrame:
    errors_mask = y_true != y_pred
    errors_idx = np.where(errors_mask)[0]
    
    sampled_idx = np.random.choice(errors_idx, size=min(len(errors_idx), num_errors), replace=False)
    
    error_records = []
    for idx in sampled_idx:
        error_records.append({
            "text": df.iloc[idx]["text"],
            "true_label": y_true[idx],
            "predicted_label": y_pred[idx]
        })
        
    error_df = pd.DataFrame(error_records)
    error_df.to_csv(save_path, index=False)
    return error_df