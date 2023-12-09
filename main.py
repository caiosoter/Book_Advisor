import streamlit as st
import re
import pickle
import torch
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def loading_data(path):
    data = pq.read_table(path).to_pandas()
    return data

@st.cache_data
def loading_embedding_bert(path):
    with open(path, 'rb') as file:
        vetor_embedding_serielizado = pickle.load(file)
    return vetor_embedding_serielizado

@st.cache_data
def filtro_categorias(titulo, dados_totais):
    categorias = [i.strip(" ") for i in re.split(r"&|,| and ", titulo["categories"].tolist()[0])]
    if len(categorias) > 1:
        filtro_categorias_multiplas = dados_totais["categories"].str.contains("|".join(categorias)) 
        data_mesma_categoria = dados_totais[filtro_categorias_multiplas]
        return data_mesma_categoria
    elif len(categorias) == 1:
        filtro_categorias_unica = dados_totais["categories"].str.contains(categorias[0]) 
        data_mesma_categoria = dados_totais[filtro_categorias_unica]
        return data_mesma_categoria 
    return None


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
data = loading_data(r"data\preprocessed_data.parquet")
embendding_matrix = loading_embedding_bert(r'data\vetor_embedding_serializado.pkl')
title_input = limpando_titulos(st.sidebar.text_input(label="Write a Title", value="Dr Seuss American icon"))
titulo_escolhido = data[data["Title_cleaned"].str.contains(r"^{}".format(title_input), regex=True)]

left_column, right_column = st.columns(2)
with st.sidebar:
    st.write(f"## What about your book?")
    if title_input and not titulo_escolhido["Title"].empty:
        author = titulo_escolhido["authors"].tolist()[0]
        year = titulo_escolhido["publishedDate"].tolist()[0]
        nome = titulo_escolhido["Title"].tolist()[0]
        categoria = titulo_escolhido["categories"].tolist()[0]
        description = titulo_escolhido["description"].tolist()[0]
        imagem = titulo_escolhido["image"].tolist()[0]
        if imagem:
            st.image(imagem)
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Categories:** {categoria}")
        st.write(f"**Published date:** {year}")
        with st.expander(label="Description"):
            st.write(f"**Description**: {description}")
    else:
        st.write("I do not have this book in my database!")


if not titulo_escolhido["Title"].empty:
    index = titulo_escolhido.index[0]
    vetor_embebending_amostra = embendding_matrix.iloc[index][0]

    # Filtrar os livros de categorias semelhantes:
    data_mesma_categoria = filtro_categorias(titulo_escolhido, data)
    similaridades_df = similaridade(embendding_matrix, vetor_embebending_amostra)
    data["similaridade"] = similaridades_df

    # Top 5 similares
    five_top_gender = data.loc[data_mesma_categoria.index].sort_values(by="similaridade", ascending=False).reset_index(drop=True).loc[1:5]
    five_top = data.sort_values(by="similaridade", ascending=False).reset_index(drop=True).loc[1:5]
    st.write(five_top_gender)
    st.write(five_top)

    





