import streamlit as st
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def loading_data(path):
    data = pq.read_table(path).to_pandas()
    #data["authors"] = data["authors"].apply(lambda x: re.sub(r"\[|\]", "", x))
    return data

@st.cache_data()
def loading_embedding():
    pass

@st.cache_resource
def loading_model():
    pass

def limpando_titulos(text):
    text = re.sub("[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r":|\(|\)|:|\[|\]|!|\?", "", text)
    text = text.strip(" ")
    text = re.sub("\s", "", text)
    text = text.lower()
    return text


st.markdown("# Book Adviser🎈")
st.subheader('I would like to suggest you a new book!!')
data = loading_data(r"data\preprocessed_data.parquet")
title_input = limpando_titulos(st.sidebar.text_input(label="Write a Title", value="Dr Seuss American icon"))
st.dataframe(data.head())

left_column, right_column = st.columns(2)
with st.sidebar:
    st.write(f"## What about your book?")
    if title_input:
        filtro = data[data["Title_cleaned"].str.contains(title_input)]
        author = filtro["authors"].tolist()[0]
        year = filtro["publishedDate"].tolist()[0]
        nome = filtro["Title"].tolist()[0]
        description = filtro["description"].tolist()[0]
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Published date:** {year}")
        with st.expander(label="Description"):
            st.write(f"**Description**: {description}")
        
        

with right_column:
    if title_input:
        filtro = data[data["Title_cleaned"].str.contains(title_input)]
        

    



