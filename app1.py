# # app.py — Prédiction du prix des véhicules 🚗
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path

# # ID du fichier sur Google Drive
# file_id = "1xz_aYsf32o9KQITtZUZ5wTMrJ7j7nlcm"  # remplace par le tien
# data_path = f"https://drive.google.com/uc?id={file_id}"

# df = pd.read_csv(data_path)



# # Supprimer les lignes vides éventuelles
# df = df.dropna(subset=["Brand", "Model", "Fuel_Type", "Transmission"])

# # Extraire les valeurs uniques de ton dataset
# brands = sorted(df["Brand"].unique())
# models = sorted(df["Model"].unique())
# fuel_types = sorted(df["Fuel_Type"].unique())
# transmissions = sorted(df["Transmission"].unique())

# # --- Configuration de la page ---
# st.set_page_config(page_title="Prédicteur de prix de véhicule", page_icon="🚗")

# # --- Chargement du modèle sauvegardé ---
# # @st.cache_resource
# # def load_model():
# #     model_path = Path("model/vehicule_price_pipeline.pkl")  # chemin du pipeline
    
# #     return joblib.load(model_path)  # Pipeline = preprocess + modèle

# # model = load_model()

# # --- Titre et description ---
# st.title("🚗 Prédiction du prix d’un véhicule")
# st.write("Remplissez les caractéristiques du véhicule pour obtenir une estimation du prix 💰")

# # # --- Options ---
# # brands = ["Toyota", "BMW", "Mercedes", "Ford", "Volkswagen", "Audi", "Hyundai", "Kia"]
# # models = ["Corolla", "Camry", "Civic", "Focus", "Golf", "A3", "Elantra", "Sportage"]
# # fuel_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
# # transmissions = ["Manual", "Automatic"]

# # --- Disposition des champs d’entrée ---
# col1, col2 = st.columns(2)
# with col1:
#     brand = st.selectbox("Marque du véhicule", brands, index=0)
#     model_name = st.selectbox("Modèle", models, index=0)
#     year = st.number_input("Année de fabrication", min_value=1990, max_value=2025, value=2018, step=1)
#     engine_size = st.number_input("Cylindrée du moteur (L)", min_value=0.5, max_value=6.0, value=1.6, step=0.1)
#     doors = st.number_input("Nombre de portes", min_value=2, max_value=5, value=4, step=1)

# with col2:
#     mileage = st.number_input("Kilométrage (km)", min_value=0, max_value=500000, value=60000, step=500)
#     fuel_type = st.selectbox("Type de carburant", fuel_types, index=0)
#     transmission = st.selectbox("Transmission", transmissions, index=0)
#     owner_count = st.number_input("Nombre de propriétaires précédents", min_value=0, max_value=10, value=1, step=1)

# # --- Construction du DataFrame pour la prédiction ---
# def build_df():
#     return pd.DataFrame([{
#         "Brand": brand,
#         "Model": model_name,
#         "Year": int(year),
#         "Engine_Size": float(engine_size),
#         "Mileage": float(mileage),
#         "Fuel_Type": fuel_type,
#         "Transmission": transmission,
#         "Doors": int(doors),
#         "Owner_Count": int(owner_count)
#     }])

# # --- Bouton de prédiction ---
# st.divider()
# if st.button("🔮 Prédire le prix"):
#     X = build_df()
#     y_pred = model.predict(X)[0]

#     st.success(f"💰 **Prix estimé du véhicule : {y_pred:,.0f} €**")

#     with st.expander("Voir les caractéristiques envoyées au modèle"):
#         st.dataframe(X, use_container_width=True)

# # --- Pied de page ---
# st.markdown(
#     """
#     <hr style="border: 1px solid #ddd;">
#     <div style='text-align: center; font-size: 15px; color: #666; margin-top: 20px;'>
#         © 2025 <b style="color:#007bff;">Lucabest</b> — Application de prédiction du prix des véhicules créée avec 🚀 Streamlit
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# app.py — Prédiction du prix des véhicules 🚗

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

# =========================
# 📊 CHARGEMENT DATASET
# =========================

file_id = "1xz_aYsf32o9KQITtZUZ5wTMrJ7j7nlcm"
data_path = f"https://drive.google.com/uc?id={file_id}"

df = pd.read_csv(data_path)

df = df.dropna(subset=["Brand", "Model", "Fuel_Type", "Transmission"])

brands = sorted(df["Brand"].unique())
models = sorted(df["Model"].unique())
fuel_types = sorted(df["Fuel_Type"].unique())
transmissions = sorted(df["Transmission"].unique())

# =========================
# ⚙️ CONFIG PAGE
# =========================

st.set_page_config(
    page_title="Prédicteur de prix de véhicule",
    page_icon="🚗"
)

# =========================
# 📦 CHARGEMENT MODELE (GOOGLE DRIVE)
# =========================

MODEL_PATH = "vehicule_price_pipeline.pkl"
FILE_ID = "1y76oxVlQH1NT3qxtL14v3CGwYg-DhT3q"

def download_model():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    response = requests.get(url)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model()

    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# 🧠 UI
# =========================

st.title("🚗 Prédiction du prix d’un véhicule")
st.write("Remplissez les caractéristiques du véhicule pour obtenir une estimation 💰")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marque du véhicule", brands)
    model_name = st.selectbox("Modèle", models)
    year = st.number_input("Année", 1990, 2025, 2018)
    engine_size = st.number_input("Cylindrée", 0.5, 6.0, 1.6)
    doors = st.number_input("Portes", 2, 5, 4)

with col2:
    mileage = st.number_input("Kilométrage", 0, 500000, 60000)
    fuel_type = st.selectbox("Carburant", fuel_types)
    transmission = st.selectbox("Transmission", transmissions)
    owner_count = st.number_input("Propriétaires", 0, 10, 1)

# =========================
# 📊 INPUT MODEL
# =========================

def build_df():
    return pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,
        "Year": int(year),
        "Engine_Size": float(engine_size),
        "Mileage": float(mileage),
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Doors": int(doors),
        "Owner_Count": int(owner_count)
    }])

# =========================
# 🔮 PREDICTION
# =========================

st.divider()

if st.button("🔮 Prédire le prix"):
    X = build_df()
    y_pred = model.predict(X)[0]

    st.success(f"💰 Prix estimé : {y_pred:,.0f} €")

    with st.expander("Voir données envoyées"):
        st.dataframe(X)

# =========================
# 🧾 FOOTER
# =========================

st.markdown(
    """
    <hr>
    <div style='text-align:center; color:gray;'>
    © 2025 Lucabest — ML Car Price Prediction 🚗
    </div>
    """,
    unsafe_allow_html=True
)