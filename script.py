import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para plotar o gráfico KS
def graph_ks(df, event_column, probability_column, event_value, non_event_value):
    score_0 = df[df[event_column] == 0][probability_column]
    score_1 = df[df[event_column] == 1][probability_column]

    x_0 = np.sort(score_0)
    y_0 = np.arange(1, len(x_0) + 1) / len(x_0)

    x_1 = np.sort(score_1)
    y_1 = np.arange(1, len(x_1) + 1) / len(x_1)

    x_all = np.sort(np.concatenate((x_0, x_1)))
    y_0_interp = np.searchsorted(x_0, x_all, side='right') / len(x_0)
    y_1_interp = np.searchsorted(x_1, x_all, side='right') / len(x_1)
    difference = np.abs(y_0_interp - y_1_interp)

    ks_statistic = np.max(difference)
    ks_index = np.argmax(difference)
    ks_x_value = x_all[ks_index]
    ks_y_value = y_0_interp[ks_index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_0, y_0, label=non_event_value, linestyle='-', color='blue')
    ax.plot(x_1, y_1, label=event_value, linestyle='-', color='orange')
    ax.annotate(f'KS = {ks_statistic:.2f}', xy=(ks_x_value, ks_y_value),
                xytext=(ks_x_value + 0.1, ks_y_value - 0.1),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.set_title('Distribuição Acumulada dos Scores por Classe com KS')
    ax.set_xlabel('SCORE')
    ax.set_ylabel('Fração Acumulada')
    ax.legend()
    ax.grid()
    return fig

# Leitura do CSV
df = pd.read_csv("customer_database_case_study_tratado.csv")

# Filtros interativos
st.sidebar.title("Filtros")

product_filter = st.sidebar.multiselect("Product Type", options=sorted(df["ProductType"].dropna().unique()), default=None)
city_filter = st.sidebar.multiselect("City", options=sorted(df["City"].dropna().unique()), default=None)
activity_filter = st.sidebar.multiselect("Economic Activity", options=sorted(df["EconomicActivity"].dropna().unique()), default=None)
person_type_filter = st.sidebar.multiselect("Person Type", options=sorted(df["PersonType"].dropna().unique()), default=None)

# Aplicar os filtros
filtered_df = df.copy()

if product_filter:
    filtered_df = filtered_df[filtered_df["ProductType"].isin(product_filter)]
if city_filter:
    filtered_df = filtered_df[filtered_df["City"].isin(city_filter)]
if activity_filter:
    filtered_df = filtered_df[filtered_df["EconomicActivity"].isin(activity_filter)]
if person_type_filter:
    filtered_df = filtered_df[filtered_df["PersonType"].isin(person_type_filter)]


event_column = "DefaultStatus"         # coluna binária: 0 ou 1
probability_column = "CreditScore"     # coluna de score
event_value = "Inadimplente"
non_event_value = "Adimplente"

# Exibir gráfico KS
if not filtered_df.empty:
    st.pyplot(graph_ks(filtered_df, event_column, probability_column, event_value, non_event_value))
else:
    st.warning("Nenhum dado encontrado com os filtros aplicados.")
