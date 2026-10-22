
# AI DataForge

**AI Data Annotation, Quality Assurance & LLM Evaluation Platform**

AI DataForge is a Python-based application for preparing datasets, reviewing annotations, evaluating AI responses, and improving data quality through structured workflows.

## Live Demo

Try the application: https://ai-dataforge-3tmqxouqfrkefayf3cb7ak.streamlit.app/

## Features

### 1. Dataset Management
- Upload CSV, JSON, and JSONL datasets.
- Preview records and inspect column data types.
- Store the active dataset in a local SQLite database.

### 2. Data Cleaning
- Trim unnecessary whitespace from text fields.
- Remove exact duplicate rows.
- Preserve missing values for review instead of deleting them automatically.

### 3. Data Quality Checks
- Identify missing values.
- Detect exact duplicate rows.
- Find repeated text values for further review.

### 4. Manual Annotation
- Assign labels to text records.
- Review annotation progress.
- Save approved annotations and export the resulting dataset.

### 5. AI-Assisted Annotation
- Generate label suggestions using the Google Gemini API.
- Restrict suggestions to a defined list of labels.
- Review or correct suggested labels before saving.
- Handle API errors and retry eligible temporary failures.

### 6. Train/Test Split
- Split datasets into training and testing sets.
- Support optional stratification and group-aware splitting.
- Check selected columns for overlapping values across splits.
- Export training and testing data separately.

### 7. LLM Evaluation
- Compare two AI-generated responses.
- Score correctness, relevance, instruction following, and clarity.
- Record reviewer preferences and generate preference datasets.

### 8. Annotation Analytics
- Track annotation progress and label distribution.
- Review available annotation quality metrics.
- Inspect AI suggestion review outcomes where recorded.

### 9. Annotation QA
- Compare labels from two annotators.
- Calculate exact agreement and Cohen's Kappa.
- Display a confusion matrix.
- Identify disagreements for review.

### 10. Annotation Adjudication
- Review conflicting labels.
- Select final labels from an approved label list.
- Save reviewed decisions to the dataset.

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- SQLite
- Google Gemini API
- Pytest

## Project Structure

```text
AI-DataForge/
├── app.py
├── app/
│   ├── __init__.py
│   ├── ai_annotation_page.py
│   ├── analytics_page.py
│   ├── annotation_qa_page.py
│   ├── adjudication_page.py
│   └── evaluation_page.py
├── core/
│   ├── __init__.py
│   ├── ai_annotation.py
│   ├── annotation.py
│   ├── annotation_qa.py
│   ├── adjudication.py
│   ├── analytics.py
│   ├── cleaning.py
│   ├── data_split.py
│   ├── evaluation.py
│   ├── quality_checks.py
│   └── storage.py
├── tests/
├── data/
│   ├── sample/
│   ├── processed/
│   └── dataforge.db
├── exports/
├── .streamlit/
│   └── secrets.toml
├── requirements.txt
└── README.md
```

The exact files may vary as the project evolves.

## Setup Instructions

### 1. Create a virtual environment

```powershell
py -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

AI-assisted annotation requires a valid API key and available API quota.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

### 6. Run automated tests

```powershell
python -m pytest -q
```

## Data Storage

Application data is stored locally in SQLite where supported by the implemented workflow.

Keep a backup of important datasets and the database. Deleting the database may remove saved application records.

## Security

- Never commit API keys or credentials.
- Keep `.streamlit/secrets.toml` out of Git.
- Do not upload private datasets to an external AI service unless authorized.
- Review AI-generated labels before treating them as final annotations.
- Evaluate data quality and potential leakage before training a model.

## Current Status

AI DataForge is an actively developed portfolio project. Features and integrations should be tested in the target environment before deployment.

## Author

Prince Parvez
