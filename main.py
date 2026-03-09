import torch

from source_code.data_loader import load_and_split_data
from source_code.preprocessing import create_dataloaders
from source_code.models import LSTMClassifier, CNNTextClassifier
from source_code.training import fit, evaluate
from source_code.evaluation import plot_learning_curves, generate_confusion_matrix, get_classification_metrics, extract_and_save_errors

def run_experiment(model_type, dropout, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names):
    print(f"\n-Running {model_type} with dropout={dropout}")
    
    if model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=len(vocab), 
            embed_dim=64, 
            dropout=dropout, 
            num_classes=4
        ).to(device)
    else:
        model = CNNTextClassifier(
            vocab_size=len(vocab), 
            embed_dim=64, 
            dropout=dropout, 
            num_classes=4
        ).to(device)

    history = fit(
        model, train_loader, dev_loader, device, 
        lr=1e-3, max_epochs=20, patience=3
    )

    file_prefix = f"outputs/{model_type}_drop{dropout}"
    
    plot_learning_curves(history, f"{file_prefix}_learning_curves.png")

    test_results = evaluate(model, test_loader, device)
    
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['acc']:.4f}")
    print(f"Test Macro-F1: {test_results['f1']:.4f}")
    
    metrics_str = get_classification_metrics(test_results["y_true"], test_results["y_pred"], class_names)
    print("\nClassification Report:\n", metrics_str)

    generate_confusion_matrix(
        test_results["y_true"], test_results["y_pred"], 
        class_names, f"{file_prefix}_confusion_matrix.png"
    )

    extract_and_save_errors(
        test_results["y_true"], test_results["y_pred"], 
        test_df, f"{file_prefix}_errors.csv", num_errors=15
    )

def main():
    torch.manual_seed(7)

    train_df, dev_df, test_df = load_and_split_data(seed=7)
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    train_loader, dev_loader, test_loader, vocab = create_dataloaders(
        train_df, dev_df, test_df, max_len=128, batch_size=64
    )

    run_experiment("cnn", 0.3, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)
    
    run_experiment("lstm", 0.3, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)
    
    run_experiment("lstm", 0.0, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)

if __name__ == "__main__":
    main()