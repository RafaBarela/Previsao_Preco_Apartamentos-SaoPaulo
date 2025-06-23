# 🏙️ Previsão de Preços de Apartamentos em São Paulo

Este projeto tem como objetivo prever o preço de apartamentos na cidade de São Paulo utilizando técnicas de Machine Learning. A aplicação foi desenvolvida com Python e possui uma interface interativa construída com Streamlit.

## 📊 Resultados do Modelo

O modelo com melhor desempenho foi o **Random Forest Regressor**, obtendo os seguintes resultados no conjunto de testes:

- **R²:** 86.05%
- **RMSE:** R$ 180.024,54
- **MAE:** R$ 85.247,39
- **MSE:** 32.408.836.018,62
- **MAPE:** 14.31%

### ✅ Validação Cruzada (K-Fold)

- **R² médio:** 86.69%
- **Desvio padrão:** 2.06%

Esses resultados indicam uma boa capacidade de generalização do modelo para novos dados.

## 🧠 Técnicas Utilizadas

- Pré-processamento de dados (tratamento de variáveis categóricas, normalização)
- Engenharia de atributos
- Modelagem com Random Forest
- Avaliação com métricas de regressão
- Validação cruzada

## 💻 Tecnologias

- Python
- Pandas, NumPy, Scikit-Learn
- Streamlit
- Matplotlib / Seaborn

## Exemplo de Predição vs Valor Real do Imóvel 

### Predição
![Image](https://github.com/user-attachments/assets/6b5b4f39-e9d1-4e88-9cfb-a3ed87497f45)

### Valor Real
![Image](https://github.com/user-attachments/assets/269394ff-c749-4aea-b781-79a5e9340283)
