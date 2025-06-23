import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Carrega o modelo e o mapa de encoding
modelo = joblib.load('modelo_randomf.joblib')
mapa_encoding = joblib.load('mapa_encoding.joblib')

# Dicionários base
x_numericos = {
    'Area(m²)': 0,
    'Banheiros': 0,
    'Suites': 0,
    'Quartos': 0,
    'Vagas': 0,
    'taxa_condominio': 0,
}

x_tf = {'Elevador': 0, 'Mobiliado': 0, 'Piscina': 0, 'Novo': 0}


# Inputs numéricos
for campo in x_numericos:
    x_numericos[campo] = st.number_input(campo.replace('_', ' ').capitalize(), min_value=0.0)
    
# Inputs binários
for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    x_tf[item] = 1 if valor == 'Sim' else 0

# Seleção de bairro
bairro_usuario = st.selectbox("Bairro", sorted(mapa_encoding.keys()))
bairro_encoded = mapa_encoding.get(bairro_usuario, np.mean(list(mapa_encoding.values())))



# Monta DataFrame para predição
dados_input = {
    **x_numericos,
    **x_tf,
    'bairro_encoded': bairro_encoded
}

df_pred = pd.DataFrame([dados_input])

# Predição
if st.button("Prever preço"):
    df_pred = df_pred[modelo.feature_names_in_]  # Garante ordem correta
    preco_predito = modelo.predict(df_pred)[0]
    st.success(f"Preço estimado: R$ {preco_predito:,.2f}")
