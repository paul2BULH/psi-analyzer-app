[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

# Enhanced Patient Safety Indicator (PSI) Analyzer

Analyze PSI 02–19 from structured inpatient claim data using advanced logic and Gemini AI support.

---

# PSI Analyzer Streamlit App

This app allows healthcare professionals and CDI specialists to evaluate patient encounters against AHRQ PSI logic (PSI 02–19).

## 📂 Files Included
- `streamlit_app.py`: Main Streamlit UI for analysis
- `PSI_02_19_Gemini_v2_Fixed.py`: PSI logic engine (class-based)
- `PSI_02_19_Compiled_Stitched_Final.json`: Code set appendix
- `PSI_Master_Input_Template_With_Disposition.xlsx`: Sample input format
- `requirements.txt`: Python dependencies

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🔍 Input Expectations
Input Excel must match the structure defined in `PSI_Master_Input_Template_With_Disposition.xlsx`.

## 📤 Output
- Filterable results table
- Inclusion/Exclusion status per PSI
- Exportable CSVs

---
Developed for internal PSI review and CDI automation workflows.
