import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the trained model
model = load_model('sawit_model.h5')

# Initialize LabelEncoders
label_encoder_jenis_tanah = LabelEncoder()
label_encoder_musim = LabelEncoder()

# Load training data
latih = pd.read_csv('data_train_sawit.csv')

# Label encode 'jenis_tanah' and 'musim' columns
latih["jenis_tanah"] = label_encoder_jenis_tanah.fit_transform(latih["jenis_tanah"])
latih["musim"] = label_encoder_musim.fit_transform(latih["musim"])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Initialize y_scaler
y_scaler = MinMaxScaler()

# Function to get user input
def get_user_input():
    tgl_panen_terakhir = st.date_input("Pilih Tanggal Terakhir Panen", min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31), value=None)
    tgl_panen = st.date_input("Pilih Tanggal Panen Berikutnya", min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31), value=None)
    umur_pilihan = list(range(0, 30)) 
    umur_tanaman = st.selectbox("Pilih Umur Tanaman", umur_pilihan)
    musim = st.selectbox("Masukkan Musim", ['hujan', 'kemarau'])
    suhu_pilihan = list(range(0, 41))  
    suhu = st.selectbox("Pilih Suhu (C)", suhu_pilihan)
   
    # Use fixed options for 'jenis_tanah'
    jenis_tanah_options = ['lempung', 'tanah berpasir', 'tanah liat']
    jenis_tanah = st.selectbox("Masukkan Jenis Tanah", jenis_tanah_options)
    jumlah_pemupukan = st.number_input("Masukkan Pemupukan (kg/ha): ")
    luas_lahan = st.number_input("Masukkan Luas Lahan (hektar): ")
    jumlah_batang = st.number_input("Masukkan Jumlah Batang: ", min_value=0, value=0, step=1)
    
    return [tgl_panen_terakhir, tgl_panen, umur_tanaman, musim, suhu, jenis_tanah, jumlah_pemupukan, luas_lahan, jumlah_batang, np.nan]

# Function to predict harvest yield
def predict_harvest(input_data):
    # Use previously fitted label_encoder_jenis_tanah to transform 'jenis_tanah'
    input_data["jenis_tanah"] = label_encoder_jenis_tanah.transform([input_data["jenis_tanah"]])[0]
    input_data["musim"] = label_encoder_musim.transform([input_data["musim"]])[0]

    # Exclude non-numeric columns from fitting process
    numeric_columns = latih.drop(['hasil_panen', 'tgl_panen_terakhir', 'tgl_panen'], axis=1).select_dtypes(include=[np.number]).columns
    scaler.fit(latih[numeric_columns])
    y_scaler.fit(latih['hasil_panen'].values.reshape(-1, 1))

    # Normalize input features using the fitted scaler
    X_test_scaled = scaler.transform(input_data[numeric_columns].values.reshape(1, -1))
    
    # Reshape data for LSTM model
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Predict harvest yield
    y_pred_scaled = model.predict(X_test_reshaped)

    # Inverse transform to get the original value
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

# Title for the Streamlit app
st.title('Prediksi Hasil Panen ')

# Display Lottie animation
st_lottie(lottie_hello, loop=True, key="hello")

# Get user input
user_input = get_user_input()

# Create DataFrame for user input
df_user_input = pd.DataFrame([user_input], columns=latih.columns)

# Button to trigger prediction
if st.button('Prediksi Hasil Panen'):
    # Predict harvest yield
    predicted_harvest = predict_harvest(df_user_input)

    # Display the predicted harvest
    st.success(f'Prediksi Hasil Panen: {predicted_harvest:.2f} Ton')
