import streamlit as st
import re
import numpy as np
import pandas as pd
import pickle
import torch
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def loading_data(path):
    data = pq.read_table(path).to_pandas()
    return data

@st.cache_data()
def loading_embedding_bert(path):
    with open(path, 'rb') as file:
        vetor_embedding_serielizado = pickle.load(file)
    return vetor_embedding_serielizado


def limpando_titulos(text):
    text = re.sub("[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r":|\(|\)|:|\[|\]|!|\?", "", text)
    text = text.strip(" ")
    text = re.sub("\s", "", text)
    text = text.lower()
    return text


def similaridade(vetor_dataframe, vetor_amostra):
    similaridades = cosine_similarity(torch.cat(vetor_dataframe.tolist()), vetor_amostra.numpy().reshape(1, -1))
    return similaridades


st.markdown("# Book Advisor :book:")
st.subheader('I would like to suggest you a new book!!')
data = loading_data(r"data\preprocessed_data.parquet").head().copy()
embendding_matrix = loading_embedding_bert(r'data\vetor_embedding_serializado.pkl')
title_input = limpando_titulos(st.sidebar.text_input(label="Write a Title", value="Dr Seuss American icon"))
st.dataframe(data.head())
filtro = data[data["Title_cleaned"].str.contains(r"^{}".format(title_input), regex=True)]

left_column, right_column = st.columns(2)
with st.sidebar:
    st.write(f"## What about your book?")
    if title_input and not filtro["Title"].empty:
        author = filtro["authors"].tolist()[0]
        year = filtro["publishedDate"].tolist()[0]
        nome = filtro["Title"].tolist()[0]
        description = filtro["description"].tolist()[0]
        imagem = filtro["image"].tolist()[0]
        if imagem:
            st.image(imagem)
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Published date:** {year}")
        with st.expander(label="Description"):
            st.write(f"**Description**: {description}")
    else:
        st.write("I do not have this book in my database!")


if not filtro["Title"].empty:
    index = filtro.index[0]
    vetor_embebending_amostra = embendding_matrix.iloc[index][0]
    similaridades_df = similaridade(embendding_matrix, vetor_embebending_amostra)
    data["similaridade"] = similaridades_df
    st.write(data)
    





