# SIG-MCDM Diagnostic Support System (2x2 Quadrant Layout)

This Streamlit app is designed to assist in the diagnosis of high-risk acute diseases using profile similarity. It uses a simple Euclidean distance-based model to compare patient symptoms against predefined disease profiles.

## 🔍 Features
- Input symptoms: fever, platelet, WBC, bleeding, fatigue, pain, nausea
- Compare against disease profiles (Sepsis, Dengue DHF, Meningitis, Leukemia)
- Generate a one-page PDF report with 2x2 layout:
  - Top-left: Patient input summary
  - Top-right: Explanation of top-ranked disease
  - Bottom-left: Radar chart (Patient vs Top Disease)
  - Bottom-right: Healthy vs Patient profile chart

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
streamlit run sig_mcdm_diagnostic_app_quadrant.py
```

## 📄 Output
- A single-page diagnosis report saved and downloadable as a PDF.
- Charts are saved automatically under `diagnosis_reports/`.

## 📁 Files
- `sig_mcdm_diagnostic_app_quadrant.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `README.md`: This documentation

Developed by Dr. Zahari Md Rodzi @ UiTM.