import streamlit as st

st.set_page_config(
    page_title="Sistem Latihan Industri",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Sistem Latihan Industri")
st.subheader("Fakulti Sains Komputer & Matematik")
st.markdown("**UiTM Cawangan Negeri Sembilan Kampus Seremban**")

st.markdown("""
Sistem ini menyokong 4 kategori pengguna:

### 👨‍🎓 Pelajar
- Isi borang BLI-01
- Jana & hantar surat SLI
- Muat naik borang BLI-02
- Isi logbook mingguan
- Muat naik laporan akhir

### 🏢 Penyelia Industri
- Log masuk untuk isi borang BLI-05 (penilaian pelajar)

### 👩‍🏫 Penyelia Akademik
- Log masuk untuk isi borang BLI-08 (penilaian akhir pelajar)

### 🧑‍💼 Penyelaras Program (CS241, CS248, CS249, CS290)
- Tetapkan penyelia akademik
- Semak kemajuan pelajar
- Muat turun keputusan akhir

Sila gunakan menu di sebelah kiri untuk navigasi ke modul masing-masing.

Sistem ini dibangunkan oleh **Dr. Zahari Md Rodzi**, UiTM Cawangan Negeri Sembilan Kampus Seremban.
""")
