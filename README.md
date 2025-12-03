# CPU-Friendly Spam Classifier (Qwen-0.5B + LoRA)

A lightweight, CPU-trainable solution for classifying SMS/Email spam using a fine-tuned Large Language Model (LLM).  
This project demonstrates how to adapt a modern SLM — **Qwen2.5-0.5B-Instruct** — for SMS spam classification using **LoRA** on standard consumer hardware.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Setup & Installation](#setup--installation)  
3. [Model Selection](#model-selection)  
4. [Dataset Details](#dataset-details)  
5. [Training Approach & Hyperparameters](#training-approach--hyperparameters)  
6. [Evaluation Results](#evaluation-results)  
7. [Inference & Guardrails](#inference--guardrails)  
8. [Limitations](#limitations)

---

## Project Overview
This tool classifies messages as **Spam** or **Ham** (legitimate).

**Core Technologies**
- Hugging Face Transformers  
- PEFT (LoRA fine-tuning)  
- CPU-friendly training & inference  

**Key Features**
- Probability-based classification  
- Natural-language reasoning  
- Safety guardrails  
- Works fully on **CPU (no GPU required)**  

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/JunYu1603/LLM-Fine-Tuning-Email-SMS-Spam-Classification.git
cd spam-classifier
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Scripts

**Train the model**
```bash
python train.py
```

**Run inference**
```bash
python inference.py
```

**Run evaluation**
```bash
python evaluate.py
```

**Run UI**
```bash
streamlit run app.py
```

**Run FastAPI**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Model Selection

### Why Qwen2.5-0.5B-Instruct?

Qwen is one of the strongest **Small Language Models (SLMs)** currently available.

Benefits of 0.5B size:
- Fits easily into RAM (<2GB)
- Runs fully on CPU
- Fast inference for real-time classification
- Can be fine-tuned in under 1 hour on a laptop CPU

---

## Dataset Details

**Source:** SMS Spam Collection Dataset (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
**Total samples:** Contains one set of SMS messages in English of 5,572 messages 
**Distribution:**  
- 4825 Ham
- 747 Spam  

### Preprocessing Steps
- Convert labels (`ham`, `spam`) → integers (`0`, `1`)
- Convert each sample into an instruction-style training prompt:

```
Classify this SMS as 'spam' or 'ham'.
SMS: <message_content>
Label: <spam/ham>
```

- Train/validation split: **80% / 20%**

---

## Training Approach & Hyperparameters

I use **LoRA (Low-Rank Adaptation)**.

Instead of modifying the entire model, only small adapter layers are trained — making CPU training fast and low-memory.

### Training Steps
1. Tokenize and pad/truncate to 128 tokens  
2. Load Qwen2.5-0.5B in full-precision (CPU-safe)  
3. Inject LoRA adapters into attention layers (`q_proj`, `v_proj`)  
4. Train model to predict the correct next-token label  
5. Save LoRA adapter weights  

### Hyperparameters

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| **Rank (r)** | 8 | Low rank works well for binary classification |
| **Alpha** | 32 | Stabilizes low-rank updates |
| **Dropout** | 0.1 | Prevents overfitting |
| **Batch Size** | 4 | Small to avoid CPU RAM issues |
| **Grad Accumulation** | 4 | Effective batch size = 16 |
| **Learning Rate** | 2e-4 | LoRA needs a higher LR |
| **Epochs** | 1 | Avoid overfitting small dataset |

---

## Evaluation Results

Evaluated using 20% validation split and 50 samples from the test set.

### 1. Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Ham** | 0.98 | 1.00 | 0.99 | 40 |
| **Spam** | 1.00 | 0.90 | 0.95 | 10 |
| **Overall Accuracy** | **~98%** | — | — | — |

### Understanding the Metrics

#### **Precision**
- **Ham Precision = 0.98**  
  → Out of all messages predicted as *Ham*, 98% were actually Ham.

- **Spam Precision = 1.00**  
  → Every message the model predicted as *Spam* was truly Spam.  
  There were **zero false positives** for spam.

#### **Recall**
- **Ham Recall = 1.00**  
  → The model correctly identified **100% of real Ham messages**.

- **Spam Recall = 0.90**  
  → The model correctly detected **9 out of 10 real Spam messages**.  
  → It missed 1 spam message (false negative).

#### **F1-Score**
- Combines Precision + Recall.
- Spam F1 = **0.95**, meaning the model performs very strongly even with only 10 spam samples in validation.


### 2. Confusion Matrix
A visual confusion matrix (`confusion_matrix.png`) is included in the repository.

---

## Inference & Guardrails

Inference uses a **Dual-Pass Strategy**.

### **Pass 1: Classification**
- LoRA adapters enabled  
- Extract logits for `"spam"` and `"ham"` tokens  
- Convert to probabilities with softmax  
- Output classification + confidence score  

### **Pass 2: Natural Language Explanation**
- LoRA adapters disabled  
- Base Qwen model generates explanation  
- Ensures fluent, accurate reasoning  

### Safety Guardrails
- Reject messages > **2000 characters**
- Block adversarial input phrases such as:
  - `ignore previous instructions`
  - `system override`
  - `delete data`

These prevent prompt injection and misuse.

---

## Limitations

1.  **Context Window (Truncation):**
    *   To ensure CPU efficiency, the model training was restricted to a max sequence length of **128 tokens**.
    *   *Impact:* Long emails will be truncated. If the "spam" indicators (like a malicious link) appear after the first 100 words, the model might miss them.

2.  **No Real-Time Verification:**
    *   The model relies purely on linguistic patterns (urgency, bad grammar, keywords).
    *   *Impact:* It cannot verify if a URL is actually malicious, check sender reputation against blacklists, or validate phone numbers, which traditional spam filters often do.

3.  **Inference Latency:**
    *   Even with a small 0.5B model, running a Transformer architecture on a CPU is significantly slower (approx. 200-500ms) compared to lightweight algorithms like Naive Bayes or Logistic Regression (<10ms).

4.  **Dataset Imbalance:**
    *   The training data contains ~87% legitimate messages and only ~13% spam.
    *   *Impact:* While precision is high, the model may be slightly biased towards classifying ambiguous messages as "safe" (False Negatives).

---

