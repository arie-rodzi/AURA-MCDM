
import io
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_excel_report(df_dict):
    """Accepts a dictionary of DataFrames with sheet names, returns Excel bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
    buffer.seek(0)
    return buffer.getvalue()

def generate_pdf_report(df, title="AURA Report"):
    """Creates a simple PDF report from one DataFrame."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, height - 50, title)

    pdf.setFont("Helvetica", 10)
    y = height - 80

    for col in df.columns:
        pdf.drawString(40, y, f"{col}")
        y -= 15

    y -= 10
    for i, row in df.iterrows():
        row_str = " | ".join(str(v) for v in row)
        if y < 50:
            pdf.showPage()
            y = height - 50
        pdf.drawString(40, y, row_str)
        y -= 15

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
