import streamlit as st
import pandas as pd
import pickle
from pycaret.regression import predict_model

# Cargar el modelo y los datos en st.session_state si no están ya cargados
if 'modelo' not in st.session_state:
    with open('best_model.pkl', 'rb') as model_file:
        st.session_state['modelo'] = pickle.load(model_file)

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

# Función para predicción individual
def prediccion_individual():
    st.header("Predicción manual de datos")
    
    # Inputs manuales
    Address = st.selectbox("Address", ['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0)
    dominio = st.selectbox("dominio", ['yahoo', 'Otro', 'gmail', 'hotmail'], index=0)
    Tec = st.selectbox("Tec", ['PC', 'Smartphone', 'Iphone', 'Portatil'], index=0)
    Avg_Session_Length = st.text_input("Avg Session Length", value="33.946241")
    Time_on_App = st.text_input("Time on App", value="10.983977")
    Time_on_Website = st.text_input("Time on Website", value="37.951489")
    Length_of_Membership = st.text_input("Length of Membership", value="3.050713")

    if st.button("Calcular predicción manual"):
        # Crear el dataframe de los inputs
        user = pd.DataFrame({
            'x0':['amarismartinezd@gmail.com'],'x1':[Address],'x2':[dominio],'x3': [Tec],
            'x4': [Avg_Session_Length], 'x5': [Time_on_App], 'x6': [Time_on_Website], 'x7':[Length_of_Membership], 'x8':[0]
        })

        # Cargar los datos de prueba y concatenar
        prueba_ = st.session_state['test_data']
        user.columns = prueba_.columns
        prueba2_ = pd.concat([user, prueba_], axis=0)
        prueba2_.index = range(prueba2_.shape[0])

        # Hacer predicciones
        predictions = predict_model(st.session_state['modelo'], data=prueba2_)

        st.write(f'La predicción es: {predictions.iloc[0]["prediction_label"]}')

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'


# Función para predicción por base de datos
def prediccion_base_datos():
    st.header("Cargar archivo para predecir")
    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

    # Botón para predecir
    if st.button("Predecir con archivo"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo subido            
                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(uploaded_file,header = 0,sep=";",decimal=",")
                else:
                    prueba = pd.read_excel(uploaded_file)
                    
#            cuantitativas = ['Avg. Session Length','Time on App','Time on Website','Length of Membership']
#            categoricas = ['Address','dominio','Tec']

                # Realizar predicción
                df_test = prueba.copy()
                predictions = predict_model(st.session_state['modelo'], data=df_test)
                predictions_label = predictions["prediction_label"]

                # Preparar archivo para descargar
                kaggle = pd.DataFrame({'Email':prueba["Email"], 'Precio': predictions["prediction_label"] })

                # Mostrar predicciones en pantalla
                st.write("Predicciones generadas correctamente!")
                st.write(kaggle)

                # Botón para descargar el archivo de predicciones
                st.download_button(label="Descargar archivo de predicciones",
                                       data=kaggle.to_csv(index=False),
                                       file_name="kaggle_predictions.csv",
                                       mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Por favor, cargue 	un archivo válido.")


    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción Académica")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()