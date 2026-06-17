import streamlit as st
import pandas as pd

@st.cache_data 
def load_data():
    df = pd.read_csv("data/french_drama_clean.csv")
    df['normalizedGenre'] = df['normalizedGenre'].fillna('Not Stated')
    df.drop('datePremiered', axis=1, inplace=True)
    #df.drop('normalizedGenre_filled', axis=1, inplace=True)
    df['yearPrinted'] = pd.to_numeric(df['yearPrinted'], errors='coerce')
    df.info()
    return df

#df=load_data()
#col = df.select_dtypes(exclude='number').columns.tolist()
#print(col)