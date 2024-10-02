import streamlit as st
import pandas as pd
from AED import Correlation
from AED import Description
from AED import histogram
from AED import Density
from AED import aberrant
from Regression import regression
from classification import classification


def choice_sidebar(dataframe,option):
    if option =="AED":

        choix2=st.sidebar.radio("",("Description","Correlation","Histogramme","Densité","Valuer Aberrantes"))
        if choix2=="Description":
            Description(dataframe)
        elif choix2=="Correlation":
            Correlation(dataframe)
        elif choix2=="Histogramme":
            histogram(dataframe)
        elif choix2=="Densité":
            Density(dataframe)
        elif choix2=="Valuer Aberrantes":
            aberrant(dataframe)
           

    elif option =="Modelisation predictive":
        column_selected = st.selectbox("Choisissez la colonne à definir comme output", dataframe.columns,index=None)
        if column_selected:
            col = dataframe.pop(column_selected)
            dataframe[column_selected] = col
            method=st.sidebar.radio("Choisir la methode",("Classification","Regression"),index=None)

            if method=="Classification":
                classification(dataframe)
            elif method=="Regression":
                regression(dataframe)


def categorielle_column_choice(dataframe,option_selected):
    categorielle=st.radio("Convertir une/plusieurs column(s) en Valuers nominale??",("Yes","No"),index=None)
    if categorielle=="Yes":
       column_selected = st.multiselect("Choisissez la/les  colonne", dataframe.columns)
       if column_selected:
        dataframe2 = pd.get_dummies(dataframe, columns=column_selected, prefix=column_selected)
        st.subheader('Raw data après conversion en valeurs nominales\n')
        st.write(dataframe2)
        choice_sidebar(dataframe,option_selected)
    elif categorielle=="No":
        st.subheader(f'Raw data\n')
        st.write(dataframe)
        choice_sidebar(dataframe,option_selected)
        

def column_choice(dataframe,option_selected):
        column = st.radio("Ya t'il le nom des columns ?",("Yes","No"),index=None)

        if column=="No":
            columns=[]
            for i in range(len(dataframe.columns)):
                new_name = st.text_input(f"Modifier le nom de la colonne {i+1}", value=dataframe.columns[i])
                columns.append(new_name)
            dataframe.columns = columns
            if st.button('Done'):
                categorielle_column_choice(dataframe,option_selected)
                
                
        elif column=="Yes":
            categorielle_column_choice(dataframe,option_selected)


def save(model,X_train,Y_train):
    import pickle
    model.fit(X_train, Y_train)
    st.write(f'Saving the model')
    model_name=st.text_input("veillez entrer le nom sur le quelle vous voulez enregistrer")
    pickle.dump(model, open(model_name, 'wb'))
    st.write(f'Model saved!')