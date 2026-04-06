# Reconciliation AI

A Streamlit-based data reconciliation tool that compares two datasets, detects matching records, flags discrepancies, and produces review-ready outputs.

## Live Demo
https://reconciliation-ai.onrender.com

## What It Does
- Suggests optimal record keys automatically
- Identifies exact, strong, and review-stage matches
- Highlights field-level differences with severity
- Flags unmatched ("orphan") records
- Exports results to CSV and Excel

## Use Cases
- Data reconciliation between systems
- KYC / client record comparison
- Operational data validation
- Audit support workflows

## Tech Stack
- Python
- Streamlit
- Pandas
- RapidFuzz
- OpenPyXL / XlsxWriter

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py