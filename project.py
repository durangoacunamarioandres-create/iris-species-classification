import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# TITULO
st.title("Clasificación de especies de iris")

# CARGAR DATASET
iris = load_iris()

# CREAR DATAFRAME
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# AGREGAR ESPECIES
df["species"] = iris.target

# CAMBIAR NUMEROS POR NOMBRES
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# MOSTRAR DATASET
st.subheader("Dataset Iris")

st.dataframe(df)

# VARIABLES
X = df.drop("species", axis=1)

y = df["species"]

# DIVIDIR DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CREAR MODELO
model = RandomForestClassifier()

# ENTRENAR MODELO
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Predicciones
y_pred = model.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

# Mostrar métricas
st.subheader("Métricas del modelo")

st.write("Accuracy:", accuracy)

st.write("Precision:", precision)

st.write("Recall:", recall)

st.write("F1 Score:", f1)

# Inputs del usuario
st.subheader("Predicción de especie")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0
)

# Botón de predicción
if st.button("Predecir especie"):

    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    st.success(
        f"La especie predicha es: {prediction[0]}"
    )

# Gráfico 3D
st.subheader("Visualización 3D del dataset Iris")

fig = px.scatter_3d(
    df,
    x='sepal length (cm)',
    y='petal length (cm)',
    z='petal width (cm)',
    color='species',
    title='Iris Dataset 3D'
)

st.plotly_chart(fig)

# Histograma
st.subheader("Distribución de variables")

fig2, ax = plt.subplots(figsize=(10, 6))

df.hist(ax=ax)

st.pyplot(fig2)