
import streamlit as st
import joblib
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import urllib.request

st.title("Bisiklet Kiralama Tahmin Uygulaması 🚲")
st.markdown("""
Bu uygulama, İzmir için canlı hava durumu verilerini kullanarak bisiklet kiralama tahminleri yapar.  
Aşağıdaki seçenekleri doldur ve tahminleri gör! 🌟
""")

# GitHub'dan dosya indir
model_url = "https://github.com/pelinsayar/peltos/blob/main/bike_rentals_model.pkl"
model_data = requests.get(model_url).content

# Modeli yükle
with open("bike_rentals_model.pkl", "wb") as f:
    f.write(model_data)

final_model = joblib.load("bike_rentals_model.pkl")


# Kullanıcıdan şehir adı alma
city = st.text_input("Şehir Adı", value="İzmir")

live_data = get_weather_data(city, forecast_type='hourly')

processed_live_data = preprocess_live_data(live_data)

predictions = final_model.predict(processed_live_data)

live_data['predicted_rentals'] = predictions

# Sonuçları ekrana yazdırma
st.write(live_data[['datetime', 'tempg', 'hum', 'weathersit', 'wind_speedg', 'predicted_rentals']])







