import streamlit as st
import pandas as pd
from Methods import column_choice

def main():
    st.write(
        """

        # Outil Automatisé d'Analyse Exploratoire des Données et de Modélisation Prédictive
        ###### cette application réalise l'Analyse Exploratoire des Données (AED) et la Modélisation Prédictive sur des jeux de données téléchargés

        """
    )
    option_selected=st.sidebar.selectbox(
        "choisir une option :",
        ("AED", "Modelisation predictive") 
        )
    
    uploaded_file=st.file_uploader("\n\nChoose a file with extension csv", type="csv")
    if uploaded_file is not None:
        
        dataframe = pd.read_csv(uploaded_file)

        column_choice(dataframe,option_selected)
    
    else:
        st.warning("Veuillez choisir un ficher csv")

if __name__ == '__main__':
    main()