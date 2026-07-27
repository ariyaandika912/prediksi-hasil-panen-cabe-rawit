import streamlit as st
import pandas as pd
import pickle


# ==========================
# Konfigurasi halaman
# ==========================

st.set_page_config(
    page_title="Prediksi Hasil Panen Cabai Rawit",
    page_icon="🌶️",
    layout="centered"
)


# ==========================
# Load Model
# ==========================

@st.cache_resource
def load_model():

    with open("model.pkl","rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ==========================
# Judul
# ==========================

st.title("🌶️ Prediksi Hasil Panen Cabai Rawit")

st.write(
"""
Sistem prediksi hasil panen cabai rawit
menggunakan algoritma Random Forest Regressor
berdasarkan faktor cuaca.
"""
)


# ==========================
# Input Data
# ==========================

tahun = st.number_input(
    "Tahun",
    min_value=2016,
    max_value=2030,
    value=2025
)


minggu = st.number_input(
    "Minggu Ke-",
    min_value=1,
    max_value=52,
    value=4
)


curah_hujan = st.number_input(
    "Curah Hujan (mm)",
    min_value=0.0,
    value=100.0
)


suhu = st.number_input(
    "Suhu (°C)",
    min_value=0.0,
    value=25.0
)


kelembapan = st.number_input(
    "Kelembapan (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)


penyinaran = st.number_input(
    "Penyinaran Jam",
    min_value=0.0,
    value=5.0
)



# ==========================
# Prediksi
# ==========================

if st.button("Prediksi"):


    data_input = pd.DataFrame({

        "Tahun":[tahun],

        "Minggu":[minggu],

        "Curah_Hujan_mm":[curah_hujan],

        "Suhu_C":[suhu],

        "Kelembapan":[kelembapan],

        "Penyinaran_Jam":[penyinaran]

    })


    hasil = model.predict(data_input)


    st.success(
        f"Hasil Prediksi Panen Cabai Rawit : {hasil[0]:,.2f} Kg"
    )
