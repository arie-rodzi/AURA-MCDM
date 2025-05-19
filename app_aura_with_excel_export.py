import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# AURA class (single file version)
class AURA:
    def __init__(self, matrix, types, weights=None, alpha=0.5, p=2):
        self.matrix = np.array(matrix, dtype=float)
        self.types = types
        self.n_criteria = self.matrix.shape[1]
        self.weights = np.array(weights if weights else [1/self.n_criteria]*self.n_criteria)
        self.alpha = alpha
        self.p = p

    def normalize(self):
        m, n = self.matrix.shape
        norm = np.zeros_like(self.matrix)
        for j in range(n):
            col = self.matrix[:, j]
            h = np.max(col) - np.min(col)
            if h == 0:
                norm[:, j] = 1.0
                continue
            if self.types[j] == 'benefit':
                k = np.max(col)
            elif self.types[j] == 'cost':
                k = np.mean(col)
            elif isinstance(self.types[j], (int, float)):
                k = self.types[j]
            else:
                raise ValueError(f"Invalid type for criterion {j}")
            norm[:, j] = 1 - np.abs(col - k) / h
        self.norm_matrix = norm

    def weighted(self):
        self.V = self.norm_matrix * self.weights

    def benchmarks(self):
        self.PIS = np.max(self.V, axis=0)
        self.NIS = np.min(self.V, axis=0)
        self.AVG = np.mean(self.V, axis=0)

    def distance(self, ref):
        return np.power(np.sum(np.abs(self.V - ref)**self.p, axis=1), 1/self.p)

    def correct(self, d):
        sigma = np.max(d) - np.min(d)
        return d + sigma * d**2

    def score(self):
        d_pos = self.correct(self.distance(self.PIS))
        d_neg = self.correct(self.distance(self.NIS))
        d_avg = self.correct(self.distance(self.AVG))
        return (self.alpha * (d_pos - d_neg) + (1 - self.alpha) * d_avg) / 2

    def rank(self):
        self.normalize()
        self.weighted()
        self.benchmarks()
        scores = self.score()
        return scores, np.argsort(scores) + 1

# Streamlit Web UI
st.set_page_config(page_title="AURA-MCDM Web App", layout="wide")
st.title("AURA (Adaptive Utility Ranking Algorithm) for MCDM")

uploaded_file = st.file_uploader("📤 Upload your Excel file (headers in row 1, weights in row 2)", type=["xlsx"])

def generate_excel_report(alternatives, criteria, weights, norm_matrix, weighted_matrix, pis, nis, avg, distances, final_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_input = pd.DataFrame(weighted_matrix, columns=criteria)
        df_input.insert(0, "Alternative", alternatives)
        df_raw = pd.DataFrame([[""] + criteria, [""] + weights] + [[a] + list(r) for a, r in zip(alternatives, weighted_matrix)])
        df_raw.to_excel(writer, sheet_name="Input Data", index=False, header=False)

        pd.DataFrame(norm_matrix, columns=criteria).to_excel(writer, sheet_name="Normalized Matrix", index=False)
        pd.DataFrame(weighted_matrix, columns=criteria).to_excel(writer, sheet_name="Weighted Matrix", index=False)

        pd.DataFrame({
            "PIS": pis,
            "NIS": nis,
            "AVG": avg
        }).to_excel(writer, sheet_name="Benchmarks", index=False)

        pd.DataFrame(distances).to_excel(writer, sheet_name="Distances", index=False)
        final_df.to_excel(writer, sheet_name="Final Ranking", index=False)

    output.seek(0)
    return output

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, header=None)
    st.subheader("📊 Raw Uploaded Data")
    st.dataframe(df_raw)

    try:
        criteria_columns = df_raw.iloc[0, 1:].tolist()
        weights = df_raw.iloc[1, 1:].astype(float).tolist()
        alternative_names = df_raw.iloc[2:, 0].tolist()
        matrix = df_raw.iloc[2:, 1:].astype(float).values

        st.subheader("🧾 Decision Matrix")
        df_matrix_display = pd.DataFrame(matrix, columns=criteria_columns)
        df_matrix_display.insert(0, "Alternative", alternative_names)
        st.dataframe(df_matrix_display)

        types = []
        for col in criteria_columns:
            if isinstance(col, str) and 'benefit' in col.lower():
                types.append('benefit')
            elif isinstance(col, str) and 'cost' in col.lower():
                types.append('cost')
            elif isinstance(col, str) and 'target' in col.lower():
                try:
                    ref = int(col.split(':')[-1].replace(')', '').strip())
                    types.append(ref)
                except:
                    types.append('benefit')
            else:
                types.append('benefit')

        if st.button("🚀 Run AURA"):
            model = AURA(matrix, types, weights)
            scores, _ = model.rank()

            sorted_indices = np.argsort(scores)
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(scores) + 1)

            df_result = pd.DataFrame({
                "Alternative": alternative_names,
                "Score": scores,
                "Rank": ranks
            }).sort_values("Rank").reset_index(drop=True)

            st.subheader("📌 Step-by-Step Results")

            with st.expander("Step 1: Normalized Decision Matrix"):
                st.dataframe(pd.DataFrame(model.norm_matrix, columns=criteria_columns))

            with st.expander("Step 2: Weighted Normalized Matrix"):
                st.dataframe(pd.DataFrame(model.V, columns=criteria_columns))

            with st.expander("Step 3: Benchmark Values"):
                st.write("Positive Ideal (PIS):", model.PIS)
                st.write("Negative Ideal (NIS):", model.NIS)
                st.write("Average Solution (AVG):", model.AVG)

            with st.expander("Step 4: Distances to Benchmarks"):
                dp = model.correct(model.distance(model.PIS))
                dn = model.correct(model.distance(model.NIS))
                da = model.correct(model.distance(model.AVG))
                df_d = pd.DataFrame({
                    "Alternative": alternative_names,
                    "Dist to PIS": dp,
                    "Dist to NIS": dn,
                    "Dist to AVG": da
                })
                st.dataframe(df_d)

            st.subheader("🏁 Final AURA Ranking (Lower Score = Better)")
            st.dataframe(df_result)

            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Result as CSV", csv, "aura_ranking.csv", "text/csv")

            excel_file = generate_excel_report(
                alternative_names, criteria_columns, weights,
                model.norm_matrix, model.V, model.PIS, model.NIS, model.AVG,
                df_d, df_result
            )
            st.download_button("⬇️ Download Full Report as Excel", excel_file, "aura_full_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.subheader("📈 AURA Score Bar Chart")
            fig, ax = plt.subplots()
            ax.barh(df_result['Alternative'], df_result['Score'], color='salmon')
            ax.invert_yaxis()
            ax.set_xlabel("Score")
            ax.set_title("AURA Final Scores (Lower is Better)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
