import pandas as pd

def check_dataset_stats():
    try:
        # Load the CSV (using latin-1 to avoid the encoding error you saw earlier)
        df = pd.read_csv("data/spam.csv", encoding="latin-1")
        
        # The dataset usually comes with columns "v1" (label) and "v2" (text)
        # We rename them to be readable
        if 'v1' in df.columns:
            df = df.rename(columns={'v1': 'label', 'v2': 'text'})
            
        # 1. Get Total Size
        total_count = len(df)
        
        # 2. Count Spam vs Ham
        label_counts = df['label'].value_counts()
        
        print("\n" + "="*30)
        print("DATASET STATISTICS")
        print("="*30)
        print(f"Total Dataset Size: {total_count} rows")
        print(f"-"*30)
        print(f"Ham (Legit):  {label_counts.get('ham', 0)} messages")
        print(f"Spam:         {label_counts.get('spam', 0)} messages")
        print("="*30 + "\n")

    except FileNotFoundError:
        print("Error: Could not find 'data/spam.csv'. Please check the file path.")

if __name__ == "__main__":
    check_dataset_stats()