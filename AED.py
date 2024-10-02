import streamlit as st
from matplotlib import pyplot
import pandas as pd


def Correlation(data):
    column_selected_cor = st.selectbox("Choisissez la colonne à afficher", data.columns,index=None)

    if column_selected_cor:
        column_selected_cor_2 = st.selectbox("Choisissez la colonne à comparer avec", data.columns,index=None)
        if column_selected_cor_2:
              st.write(f"# Correlation pour la colonne {column_selected_cor}\n")
              st.dataframe(pd.DataFrame(data[[column_selected_cor,column_selected_cor_2]]).corr())
              dt = pd.DataFrame(data[[column_selected_cor,column_selected_cor_2]]).corr()
              missing(dt)
    else:
       all_col=st.toggle("Afficher toute les colonne")
       if all_col:
         st.write("# Correlation\n")
         st.dataframe(data.corr())
         dt= data.corr()
         missing(dt)
              

def Description(data):
    column_selected_des = st.selectbox("Choisissez la colonne à afficher", data.columns,index=None)
    
    if column_selected_des:
       st.write(f"# Descriptive Statistics pour la colonne {column_selected_des}\n")
       st.dataframe(data[column_selected_des].describe())
       dt= data[column_selected_des].describe()
       missing(dt)
    else:
       all_col=st.toggle("Afficher toute les colonne")
       if all_col:
              st.write("# Descriptive Statistics\n")
              st.dataframe(data.describe())
              dt= data.describe()
              missing(dt)

def histogram(data):
    column_selected_his = st.selectbox("Choisissez la colonne à afficher", data.columns,index=None)
    if column_selected_his:
       st.write(f"# Histogramme pour la colonne {column_selected_his}\n")
       fig = pyplot.figure(figsize=(9, 9))
       data[column_selected_his].hist(ax=fig.gca())
       st.pyplot(fig)
    else:
       all_col=st.toggle("Afficher toute les colonne")
       if all_col:
              st.write("# Histogramme\n")
              fig=pyplot.figure(figsize=(9, 9))
              data.hist(ax=fig.gca())
              st.pyplot(fig)

def Density(data):
    column_selected_den = st.selectbox("Choisissez la colonne à afficher", data.columns,index=None)
    if column_selected_den:
       st.write(f"# Densité pour la colonne {column_selected_den}\n")
       fig = pyplot.figure(figsize=(13, 11))
       data[column_selected_den].plot(kind='density', subplots=True, layout=(1, 1), sharex=False, figsize=(13, 11,), ax=fig.gca())
       st.pyplot(fig)

    else:
      all_col=st.toggle("Afficher toute les colonne")
      if all_col:
       st.write("# Densité\n")
       fig=pyplot.figure(figsize=(13, 11))
       data.plot(kind='density', subplots=True, layout=(4, 4), sharex=False, figsize=(13, 11,),ax=fig.gca())
       st.pyplot(fig)

def aberrant(data):
    column_selected_abr = st.selectbox("Choisissez la colonne à afficher", data.columns,index=None)
    if column_selected_abr:
       st.write(f"# Valeurs Aberrantes pour la colonne {column_selected_abr}\n")
       fig = pyplot.figure(figsize=(10, 9))
       data[column_selected_abr].plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(13, 11), ax=fig.gca())
       st.pyplot(fig)
    else:
      all_col=st.toggle("Afficher toute les colonne")
      if all_col:
       st.write("# Valuer Aberrantes\n")
       fig=pyplot.figure(figsize=(10, 9))
       data.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(13, 11), ax=fig.gca())
       st.pyplot(fig)

def missing(data):
    missing=st.toggle("Missing values?")
    if missing:
        choix2=st.radio("",("bfill","ffill","interpolate"))
        if choix2=="bfill":
            st.dataframe(data.bfill())
        elif choix2=="ffill":
            st.dataframe(data.ffill())
        elif choix2=="interpolate":
            st.dataframe(data.interpolate())