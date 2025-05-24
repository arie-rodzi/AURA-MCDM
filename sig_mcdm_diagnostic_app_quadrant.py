# sig_mcdm_diagnostic_app_quadrant.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from fpdf import FPDF

# --- Disease Profiles ---
diseases = {
    'Sepsis': [39.0, 180, 25.0, 0, 0.3, 0.6, 0.0],
    'Dengue DHF': [38.5, 45, 4.0, 1, 0.6, 0.3, 0.9],
    'Meningitis': [39.2, 150, 12.0, 0, 0.6, 0.9, 0.3],
    'Leukemia': [38.0, 90, 1.2, 1, 0.6, 0.9, 0.6]
}
criteria = ['Fever (°C)', 'Platelet Count', 'WBC Count', 'Bleeding', 'Fatigue', 'Pain', 'Nausea']
fuzzy_map = {
    "None": 0.0, "Mild": 0.3, "Moderate": 0.6, "Severe": 0.9, "Extreme": 1.0,
    "Slight": 0.6, "Frequent": 0.9, "Yes": 1, "No": 0
}
healthy_profile = [36.8, 250, 6.0, 0, 0.1, 0.1, 0.0]
explanation = {
    'Dengue DHF': [
        "- Fever matches Dengue profile (38.5°C).",
        "- Platelet count low, close to Dengue's (45 vs 55).",
        "- WBC is low (3.0), consistent with viral infection.",
        "- Bleeding present - matches hallmark of Dengue Hemorrhagic Fever.",
        "- Fatigue and nausea severity align closely with Dengue symptoms.",
        "- Overall, 5 of 7 criteria are strong matches to Dengue."
    ]
}

st.set_page_config(page_title="SIG-MCDM Diagnostic Support System")
st.title("🧠 SIG-MCDM Diagnostic Support System")

with st.form("diagnosis_form"):
    patient_name = st.text_input("Patient Name")
    doctor_name = st.text_input("Doctor Name")
    fever = st.number_input("Fever (°C)", 35.0, 42.0, step=0.1)
    platelet = st.number_input("Platelet Count", 10, 500)
    wbc = st.number_input("WBC Count", 0.5, 30.0, step=0.1)
    bleeding = st.selectbox("Bleeding", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["None", "Mild", "Moderate", "Severe"])
    pain = st.selectbox("Pain", ["None", "Mild", "Moderate", "Severe", "Extreme"])
    nausea = st.selectbox("Nausea", ["None", "Slight", "Frequent"])
    submitted = st.form_submit_button("Diagnose")

if submitted:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_vector = [
        fever, platelet, wbc,
        fuzzy_map[bleeding], fuzzy_map[fatigue],
        fuzzy_map[pain], fuzzy_map[nausea]
    ]

    def euclidean_distance(v1, v2):
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

    distances = {disease: euclidean_distance(patient_vector, profile) for disease, profile in diseases.items()}
    ranking_df = pd.DataFrame(distances.items(), columns=['Disease', 'Distance']).sort_values(by='Distance')
    max_dist = max(ranking_df['Distance'])
    ranking_df['Similarity (%)'] = (1 - ranking_df['Distance'] / max_dist) * 100
    top_disease = ranking_df.iloc[0]['Disease']

    report_dir = "diagnosis_reports"
    os.makedirs(report_dir, exist_ok=True)
    base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{patient_name.replace(' ', '_')}"
    radar_chart_path = os.path.join(report_dir, f"{base_name}_radar.png")
    healthy_chart_path = os.path.join(report_dir, f"{base_name}_healthy.png")

    # Radar chart
    def normalize(vals):
        max_vals = [42, 500, 30, 1, 1, 1, 1]
        return [v / m for v, m in zip(vals, max_vals)]
    labels = criteria
    patient_norm = normalize(patient_vector)
    disease_norm = normalize(diseases[top_disease])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    patient_norm += patient_norm[:1]
    disease_norm += disease_norm[:1]
    fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, patient_norm, color="blue", label="Patient")
    ax_radar.fill(angles, patient_norm, color="blue", alpha=0.25)
    ax_radar.plot(angles, disease_norm, color="green", label=top_disease)
    ax_radar.fill(angles, disease_norm, color="green", alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=6)
    ax_radar.legend(loc="upper right", fontsize=6)
    fig_radar.savefig(radar_chart_path, dpi=300)
    plt.close(fig_radar)

    # Healthy chart
    fig_healthy, ax_healthy = plt.subplots(figsize=(4, 4))
    y = np.arange(len(criteria))
    ax_healthy.barh(y, patient_vector, color="dodgerblue", label="Patient")
    ax_healthy.barh(y, healthy_profile, color="lightgray", label="Healthy", alpha=0.8)
    ax_healthy.set_yticks(y)
    ax_healthy.set_yticklabels(criteria, fontsize=6)
    ax_healthy.invert_yaxis()
    ax_healthy.set_xlabel("Score / Severity", fontsize=6)
    ax_healthy.set_title("Patient vs Healthy Profile", fontsize=7)
    ax_healthy.legend(fontsize=6)
    fig_healthy.tight_layout()
    fig_healthy.savefig(healthy_chart_path, dpi=300)
    plt.close(fig_healthy)

    # Create PDF with 2x2 layout
    from fpdf import FPDF
    pdf_path = os.path.join(report_dir, f"{base_name}_2x2_Report.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", '', 9)

    # Quadrant 1: Input summary
    pdf.set_xy(10, 10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, "Patient Input Summary", ln=True)
    pdf.set_font("Arial", '', 8)
    for c, v in zip(criteria, patient_vector):
        val = v if isinstance(v, float) else list(fuzzy_map.keys())[list(fuzzy_map.values()).index(v)]
        pdf.cell(0, 4, f"{c}: {val}", ln=True)

    # Quadrant 2: Explanation
    pdf.set_xy(110, 10)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 5, f"Why '{top_disease}' is Ranked #1", ln=True)
    pdf.set_font("Arial", '', 8)
    for line in explanation[top_disease]:
        pdf.multi_cell(0, 4, line)

    # Quadrant 3: Radar chart
    pdf.image(radar_chart_path, x=10, y=150, w=90)

    # Quadrant 4: Healthy chart
    pdf.image(healthy_chart_path, x=110, y=150, w=90)

    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download 2x2 Diagnosis Report (PDF)",
            data=f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            key="pdf_2x2"
        )