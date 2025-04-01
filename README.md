# 🧪 LOHC Molecular Generator with LLM & ML Filtering

This Python script automates the **generation, evaluation, and filtering** of novel Liquid Organic Hydrogen Carrier (LOHC) molecules. It combines a **Large Language Model (LLM)** for molecular ideation with **machine learning models** and chemistry heuristics for property-based screening.

------

## 🔍 What It Does

1. **Loads Initial SMILES** from a CSV file (e.g.: `expt_31.csv`)
2. **Calls an LLM API** to generate batches of new candidate molecules in SMILES format
3. **Validates and Filters Molecules** based on:
   - Canonical validity (via RDKit)
   - Hydrogen storage potential (≥ 5.5% H₂ by weight)
   - Machine learning-predicted ΔH (between 40–70)
   - Melting point (≤ 40°C)
4. **Iteratively Expands the Set** until a desired number of candidates is reached
5. **Exports Results** to a CSV file (default: `Argo_LOHC_generated_dry_run_withMP_more_attempts_second_try_expt.csv`)

------

## 📦 Dependencies

Install the following packages before running the script:

```bash
pip install pandas numpy rdkit requests joblib leruli
```

------

## 🧬 Input Files

- `expt_31.csv` – Initial set of SMILES strings (first column is read)
- `api.txt` – A file containing the API URL for LLM-based molecular generation
- `QM9-LOHC-RF.joblib` – Trained Random Forest model for predicting ΔH (not included here due to size limit, available upon request)

------

## 🚀 Usage

Run the script with:

```bash
python LLM_LOHC_generate_final.py
```

Make sure `api.txt`, `QM9-LOHC-RF.joblib`, and the CSV file exist in the same directory.

------

## ⚙️ Configuration

Modify these parameters at the top of the script:

```python
INITIAL_SET_SIZE = 31
MAX_ITERATIONS = 10
TARGET_COUNT = 200
NEW_LOHC_BATCH_SIZE = 30
MAX_LLM_ATTEMPTS = 10
```

------

## 📤 Output

The script creates:

- A **log file**: `output_expt.log` containing real-time stdout/stderr
- A **CSV file**: with filtered SMILES and their predicted properties

------

## 🧠 Filtering Criteria

Each generated molecule is filtered based on:

| Criterion               | Requirement            |
| ----------------------- | ---------------------- |
| Valid SMILES            | Canonical RDKit SMILES |
| H₂ Weight %             | ≥ 5.5%                 |
| ΔH (via ML)             | Between 40 and 70      |
| Melting Point (via API) | ≤ 40°C or unknown      |

------

## 📈 Model Details

- ΔH predictions use a **Random Forest** trained on QM9-like molecular fingerprints.
- Fingerprints are computed using **Morgan fingerprints** with RDKit.

------

## 🧪 Hydrogen Storage Calculation

The script estimates hydrogen weight percent and number of transferable H₂ units by simulating hydrogenation (replacing unsaturations and heteroatoms).

------

## 🤖 API Notes

This workflow was written using Argo, an Argonne internal interface to OpenAI ChatGPT. Make sure to tailor the code to work with the LLM of your choice.
