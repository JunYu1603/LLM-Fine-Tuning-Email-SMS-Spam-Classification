import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from datasets import load_dataset
from inference import classify_email
from tqdm import tqdm

def evaluate_model():
    # Load a small slice of test data
    dataset = load_dataset("csv", data_files="data/spam.csv", split="train", encoding="latin-1")
    
    if "v1" in dataset.column_names and "v2" in dataset.column_names:
        dataset = dataset.rename_columns({"v1": "label", "v2": "sms"})
    
    dataset = dataset.map(lambda x: {"label": 1 if x["label"] == "spam" else 0})

    dataset = dataset.train_test_split(test_size=0.2)["test"]
    # Take just 50 samples to save time on CPU inference
    subset = dataset.select(range(50)) 
    
    true_labels = []
    pred_labels = []

    print("Running evaluation on 50 samples...")
    for item in tqdm(subset):
        text = item['sms']
        true_label = "spam" if item['label'] == 1 else "ham"
        
        # Run inference
        result = classify_email(text)
        
        true_labels.append(true_label)
        pred_labels.append(result['label'])

    # Calculate Metrics
    acc = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=["ham", "spam"])
    
    # Plotting (Optional, saves to file)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()