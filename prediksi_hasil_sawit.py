import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime


st.set_page_config(
    page_title="Prediksi Hasil Sawit",
    page_icon="logo_sawit.png",  
)

# Load the trained model
model = load_model('sawit_model new.h5')
# Load training data
train = pd.read_csv('data_sawit.csv')

# class LabelEncoder
class LabelEncoder:
    def __init__(self):
        self.mapping = {}

    def fit_transform(self, kolom):
        nilai_unik = kolom.unique()
        pemetaan = {nilai: idx for idx, nilai in enumerate(nilai_unik)}
        self.mapping[kolom.name] = pemetaan
        return kolom.map(pemetaan)

    def transform(self, kolom):
        pemetaan = self.mapping.get(kolom.name)
        if pemetaan:
            return kolom.map(pemetaan)
        else:
            raise ValueError(f"Tidak ditemukan pemetaan untuk kolom '{kolom.name}'")

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()

# Lakukan label encoding pada kolom "Jenis Tanah" dan "Musim"
train["jenis_tanah"] = label_encoder.fit_transform(train["jenis_tanah"])
train["musim"] = label_encoder.fit_transform(train["musim"])

# Memisahkan fitur dan label
X = train.drop(['hasil_panen', 'tgl_panen_terakhir', 'tgl_panen'], axis=1).values
y = train['hasil_panen'].values
# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalisasi data (menggunakan MinMaxScaler untuk seluruh fitur)
X_scaled = scaler.fit_transform(X)
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Function to get user input
def get_user_input():
    tgl_panen_terakhir = st.date_input("Pilih Tanggal Terakhir Panen", min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31), value=None)
    tgl_panen = st.date_input("Pilih Tanggal Panen Berikutnya", min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31), value=None)
    umur_pilihan = list(range(6, 30)) 
    umur_tanaman = st.selectbox("Pilih Umur Tanaman", umur_pilihan)
    musim = st.selectbox("Masukkan Musim", ['hujan', 'kemarau'])
    jenis_tanah_options = ['lempung', 'tanah berpasir', 'tanah liat']
    jenis_tanah = st.selectbox("Masukkan Jenis Tanah", jenis_tanah_options)
    jumlah_pemupukan = st.number_input("Masukkan Pemupukan (kg/ha): ")
    luas_lahan = st.number_input("Masukkan Luas Lahan (hektar): ")
    jumlah_batang = st.number_input("Masukkan Jumlah Batang: ", min_value=0, value=0, step=1)
    
    return [tgl_panen_terakhir, tgl_panen, umur_tanaman, musim, jenis_tanah, jumlah_pemupukan, luas_lahan, jumlah_batang]


def prediksi_sawit(user_input):
    # Inisialisasi LabelEncoder
    label_encoder = LabelEncoder()

    # Lakukan label encoding pada kolom "Jenis Tanah" dan "Musim"
    df_user_input["jenis_tanah"] = label_encoder.fit_transform(df_user_input["jenis_tanah"])
    df_user_input["musim"] = label_encoder.fit_transform(df_user_input["musim"])

    # Mengambil fitur dari dataset test
    X_test = df_user_input.drop(['tgl_panen_terakhir', 'tgl_panen'], axis=1).values

    # Normalisasi fitur dataset test menggunakan skaler yang sama
    X_test_scaled = scaler.transform(X_test)

    # Reshape data untuk sesuai dengan input model
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    # Memprediksi nilai Hasil_Panen/Ton
    y_pred_scaled = model.predict(X_test_reshaped)

    # Invers transformasi untuk mendapatkan nilai asli
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    return y_pred[0][0]

# Streamlit App
import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/ccafe4ff-6ca7-4467-919d-54d5601b3c19/XPReODK038.json"
lottie_hello = load_lottieurl(lottie_url_hello)

# Custom Theme
custom_style = """
    <style>
        .stButton>button {
            background-color: #294B29 !important;
            color: white !important;
            border-radius: 5px !important;
        }
        .stSuccess {
            color: white !important;
            background-color: #294B29 !important;
            border-radius: 15px !important;
            padding: 10px;
            font-weight: bold;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        h1 {
            text-align: center !important; /* CSS untuk memusatkan judul */
        }
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Title 
st.title('Prediksi Hasil Panen Sawit')


# menampilkan animation
st_lottie(lottie_hello, loop=True, key="hello")

# mendapatkan user input
user_input = get_user_input()

# membuat DataFrame user input
df_user_input = pd.DataFrame([user_input], columns=['tgl_panen_terakhir', 'tgl_panen', 'umur_tanaman', 'musim', 'jenis_tanah', 'jumlah_pemupukan', 'luas_lahan', 'jumlah_batang'])

# Button untuk melakukan prediksi
if st.button('Prediksi Hasil Panen'):
    
    predicted_sawit = prediksi_sawit(user_input)

    
    st.markdown(f'<p class="stSuccess">Prediksi Hasil Panen: {predicted_sawit:.2f} Ton</p>', unsafe_allow_html=True)
