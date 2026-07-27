import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


# membaca dataset
data = pd.read_csv(
    "Dataset_Cabe_Rawit_Baros_2016_2025.csv",
    sep=";"
)


# rapikan nama kolom
data.columns = data.columns.str.strip()


# ubah nama kolom
data = data.rename(columns={
    "Tahun":"tahun",
    "Minggu":"minggu",
    "Curah_Hujan_mm":"curah_hujan",
    "Suhu_C":"suhu",
    "Kelembapan_%":"kelembapan",
    "Penyinaran_Jam":"penyinaran",
    "Hasil_Panen_Kg":"hasil_panen"
})


# hapus data kosong
data = data.dropna()


# fitur
X = data[
[
    "tahun",
    "minggu",
    "curah_hujan",
    "suhu",
    "kelembapan",
    "penyinaran"
]
]


# target
y = data["hasil_panen"]


# model Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# training
model.fit(X,y)


# simpan model
joblib.dump(
    model,
    "model.pkl"
)


print("MODEL BERHASIL DISIMPAN")
