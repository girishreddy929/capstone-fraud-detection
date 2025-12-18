# Capstone Fraud Detection – Explainable AI Pipeline

## Overview

This repository implements an **end-to-end fraud detection explanation system** designed as a capstone project. The solution goes beyond model scoring and focuses on **explainability, transparency, reporting, and human feedback**, aligned with real-world risk, audit, and compliance requirements.

The pipeline covers **Tasks 1–7**, including:

* Data validation
* Rule-based reasoning
* LLM-generated explanations
* SHAP-based feature attribution
* Reporting & visualization
* Human evaluation and feedback loop

The final output is a **business-ready dataset and reports** that explain *why* a transaction is flagged as fraud.

---

## Project Structure

```
capstone-fraud-detection/
│
├── data/
│   ├── raw/                    # Input fraud model outputs
│   └── processed/              # Final explained dataset
│
├── reports/                    # Generated charts and visual reports
│
├── src/
│   ├── data_loader/            # Task 1: Load & validate data
│   ├── explanation/            # Tasks 2–4: Rules, LLM, SHAP
│   ├── data_process/           # Tasks 6–7: Reporting & feedback
│   ├── final_dataset/          # Task 5: Final explained dataset
│   └── final/                  # Orchestration (run all tasks)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tasks Breakdown (What Does What)

### **Task 1 – Data Loading & Validation**

**Location:** `src/data_loader/load_fraud_output.py`

* Loads raw fraud model output CSV
* Validates schema and required columns
* Derives missing flags (geo mismatch, velocity, etc.)
* Ensures clean, analysis-ready input

---

### **Task 2 – Rule-Based Reasoning**

**Location:** `src/explanation/templates.py`

* Applies deterministic fraud rules such as:

  * High transaction amount
  * Geo mismatch
  * Velocity risk
  * Device fingerprint change
* Produces human-readable **rule-based factors**
* Ensures explainability even without LLM availability

---

### **Task 3 – LLM-Based Narrative Explanations**

**Location:** `src/explanation/llm_narrative_openai.py`

* Uses OpenAI API to generate natural-language explanations
* Converts fraud signals into SME-friendly narratives
* Designed to be auditable, concise, and action-oriented

> Requires `OPENAI_API_KEY` set as an environment variable

---

### **Task 4 – SHAP Feature Attribution**

**Location:** `src/explanation/shap_integration.py`

* Computes SHAP values for the fraud model
* Extracts top contributing features per transaction
* Enables model transparency and regulatory compliance

---

### **Task 5 – Final Explained Dataset**

**Location:** `src/final_dataset/final_explained_dataset.py`

* Combines:

  * Raw model outputs
  * Rule-based factors
  * LLM explanations
  * SHAP top features
* Produces a single **business-consumable dataset**

Output:

```
data/processed/fraud_model_processed.csv
```

---

### **Task 6 – Reports & Visualizations**

**Location:** `src/data_process/vizualization_reporting.py`

Generates visual artifacts including:

* Fraud score distribution
* Fraud vs non-fraud counts
* Rule-based factor frequency
* SHAP top feature distributions

Outputs saved to:

```
reports/
```

---

### **Task 7 – Human Evaluation & Feedback Loop**

**Location:** `src/data_process/feedback_system.py`

* Allows SMEs to rate explanations on:

  * Clarity
  * Accuracy
  * Actionability
* Captures structured feedback
* Calculates success metrics (≥ 85% clarity/usefulness)
* Designed to refine:

  * Prompt templates
  * LLM instructions
  * Feature rules

---

## How to Execute (End-to-End)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/girishreddy929/capstone-fraud-detection.git
cd capstone-fraud-detection
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set OpenAI API Key (Required for Task 3)

```bash
setx OPENAI_API_KEY "your_api_key_here"   # Windows
# export OPENAI_API_KEY="your_api_key_here"  # Mac/Linux
```

---

### 5️⃣ Run Entire Pipeline (Tasks 1–7)

```bash
python src/final/run_all_tasks.py
```

This will:

* Validate data
* Generate explanations
* Compute SHAP features
* Produce final dataset
* Create reports
* Collect SME feedback

---

## Key Outputs

| Output                  | Location                                   |
| ----------------------- | ------------------------------------------ |
| Final explained dataset | `data/final/fraud_explainability.csv` |
| Visual reports          | `reports/`                                 |
| SME feedback            | CSV/JSON (feedback system output)          |

---

## Design Principles

* Explainability-first
* SME-readable outputs
* Audit & compliance ready
* Modular and extensible
* Real-world fraud workflow alignment

---

## Future Enhancements

* UI for SME feedback capture
* Model retraining using feedback
* Multi-model explanation comparison
* Role-based explanation views

---
