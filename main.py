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

def plotar_dados(df):
        author = df["authors"].tolist()[0]
        year = df["publishedDate"].tolist()[0]
        nome = df["Title"].tolist()[0]
        categoria = df["categories"].tolist()[0]
        description = df["description"].tolist()[0]
        imagem = df["image"].tolist()[0]
        if imagem:
            st.image(imagem)
        else:
            st.image("livro_comum.jpg")
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Categories:** {categoria}")
        st.write(f"**Published date:** {year}")
        with st.expander(label="Description"):
            st.write(f"**Description**: {description}")


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
embendding_matrix = loading_embedding_bert(r'data\vetor_embedding_bert_large.pkl')
title_input = limpando_titulos(st.sidebar.text_input(label="Write a Title", value="Dr Seuss American icon"))
titulo_escolhido = data[data["Title_cleaned"].str.contains(r"^{}".format(title_input), regex=True)]

left_column, middle_column, right_column = st.columns(3, gap="large")
with st.sidebar:
    st.write(f"## What about your book?")
    if title_input and not titulo_escolhido["Title"].empty:
        plotar_dados(titulo_escolhido)
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
    five_top_gender = data.loc[data_mesma_categoria.index].sort_values(by="similaridade", ascending=False).reset_index(drop=True).loc[1:]
    five_top = data.sort_values(by="similaridade", ascending=False).reset_index(drop=True).loc[1:]
    if len(five_top_gender) > 0:
        with left_column:
            primeiro_livro = five_top_gender.loc[[1]]
            plotar_dados(primeiro_livro)
        
        with middle_column:
            segundo_livro = five_top_gender.loc[[2]]
            plotar_dados(segundo_livro)

        with right_column:
            terceiro_livro = five_top_gender.loc[[3]]
            plotar_dados(terceiro_livro)   
    else:
        st.write("## There are no books of the same category!!")

    st.subheader("Books of diferent categories, that could be of your interest.")
    left_column_2, middle_column_2, right_column_2 = st.columns(3, gap="large")
    with left_column_2:
        primeiro_livro_similarity = five_top.loc[[1]]
        if primeiro_livro["Title"].values[0] == primeiro_livro_similarity["Title"].values[0]:
            primeiro_livro_similarity = five_top.loc[[4]]
        plotar_dados(primeiro_livro_similarity)

    with middle_column_2:
        segundo_livro_similarity = five_top.loc[[2]]
        if segundo_livro["Title"].values[0] == segundo_livro_similarity["Title"].values[0]:
            segundo_livro_similarity = five_top.loc[[5]]
        plotar_dados(segundo_livro_similarity)
    
    with right_column_2:
        terceiro_livro_similarity = five_top.loc[[3]]
        if terceiro_livro["Title"].values[0] == terceiro_livro_similarity["Title"].values[0]:
            terceiro_livro_similarity = five_top.loc[[6]]
        plotar_dados(terceiro_livro_similarity)

    st.write(data.sample(n=30))

    





