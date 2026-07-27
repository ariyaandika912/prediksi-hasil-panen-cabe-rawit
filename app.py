import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Panen Cabe Rawit",
    page_icon="🌶️",
    layout="wide"
)


# load model
@st.cache_resource
def load_model():

    model = joblib.load(
        "model.pkl"
    )

    return model


model = load_model()


# judul
st.title(
    "🌶️ Prediksi Hasil Panen Cabe Rawit"
)


st.write(
    "Implementasi Algoritma Random Forest Regressor"
)


st.divider()


# input pengguna

col1,col2 = st.columns(2)


with col1:

    tahun = st.number_input(
        "Tahun",
        2021,
        2030,
        2025
    )


    minggu = st.number_input(
        "Minggu",
        1,
        52,
        1
    )


    curah_hujan = st.number_input(
        "Curah Hujan (mm)",
        0.0,
        1000.0,
        200.0
    )


with col2:

    suhu = st.number_input(
        "Suhu (°C)",
        10.0,
        40.0,
        25.0
    )


    kelembapan = st.number_input(
        "Kelembapan (%)",
        0.0,
        100.0,
        80.0
    )


    penyinaran = st.number_input(
        "Penyinaran (Jam)",
        0.0,
        15.0,
        6.0
    )


# tombol prediksi

if st.button("🔍 Prediksi Hasil Panen"):


    data_input = pd.DataFrame({

        "tahun":[tahun],
        "minggu":[minggu],
        "curah_hujan":[curah_hujan],
        "suhu":[suhu],
        "kelembapan":[kelembapan],
        "penyinaran":[penyinaran]

    })


    hasil = model.predict(
        data_input
    )


    st.success(
        f"Prediksi Hasil Panen : {hasil[0]:,.2f} Kg"
    )



# grafik dataset

st.divider()

st.subheader(
    "📈 Grafik Data Hasil Panen"
)


try:

    df = pd.read_csv(
        "Dataset_Cabe_Rawit_Baros_2016_2025.csv",
        sep=";"
    )


    df.columns = df.columns.str.strip()


    fig, ax = plt.subplots()


    ax.plot(
        df["Minggu"],
        df["Hasil_Panen_Kg"]
    )


    ax.set_xlabel(
        "Minggu"
    )


    ax.set_ylabel(
        "Hasil Panen Kg"
    )


    st.pyplot(fig)


except:

    st.warning(
        "Grafik tidak tersedia"
    )
