"""
In this file, we combine functionality from all other scripts. No new funtionality was introduced.
"""
import torch

from source_code.data_loader import load_and_split_data
from source_code.preprocessing import create_dataloaders
from source_code.models import LSTMClassifier, CNNTextClassifier
from source_code.training import fit, evaluate
from source_code.evaluation import plot_learning_curves, generate_confusion_matrix, get_classification_metrics, extract_and_save_errors

def run_experiment(model_type, dropout, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names):
    #A helper function that handles a full model run
    print(f"\n-Running {model_type} with dropout={dropout}")
    
    #Initialize the right model architecture
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
            dropout=dropout, #Passed an arg for ablation
            num_classes=4
        ).to(device) #Move model to GPU / CPU

    #Train model, including early stopping
    history = fit(
        model, train_loader, dev_loader, device, 
        lr=1e-3, max_epochs=20, patience=3
    )

    file_prefix = f"outputs/{model_type}_drop{dropout}" #Create unique prefix
    
    plot_learning_curves(history, f"{file_prefix}_learning_curves.png") #save learning curves

    test_results = evaluate(model, test_loader, device) 
    
    #Print metrics and classfication report
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['acc']:.4f}")
    print(f"Test Macro-F1: {test_results['f1']:.4f}")
    

    metrics_str = get_classification_metrics(test_results["y_true"], test_results["y_pred"], class_names)
    print("\nClassification Report:\n", metrics_str)

     #save confusion matrix
    generate_confusion_matrix(
        test_results["y_true"], test_results["y_pred"], 
        class_names, f"{file_prefix}_confusion_matrix.png"
    )


    #extract and save missclassified examples
    extract_and_save_errors(
        test_results["y_true"], test_results["y_pred"], 
        test_df, f"{file_prefix}_errors.csv", num_errors=15
    )

def main():
    torch.manual_seed(7) #Seed for reproducible weight initialization

    #We ran code using T4 GPU on Google Collab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, dev_df, test_df = load_and_split_data(seed=7)     #Load data and apply a 90/10 split
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    #Build vocabularly from training data and wrap in dataloaders
    train_loader, dev_loader, test_loader, vocab = create_dataloaders(
        train_df, dev_df, test_df, max_len=128, batch_size=64
    )
    #Baseline CNN
    run_experiment("cnn", 0.3, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)
    
    #Baseline LSTM
    run_experiment("lstm", 0.3, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)
    
    #Ablation Study (Dropout 0.3 to 0.0)
    run_experiment("lstm", 0.0, train_loader, dev_loader, test_loader, vocab, device, test_df, class_names)

if __name__ == "__main__":
    main()