import streamlit as st
import re
import joblib
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
            st.image(st.secrets["link_sem_imagem"])
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Categories:** {categoria}")
        st.write(f"**Published date:** {year}")
        with st.expander(label="Description"):
            st.write(f"**Description**: {description}")


def limpando_titulos(text):
    text = re.sub("[^a-zA-Z0-9\s]", "", text)
    text = text.strip(" ")
    text = re.sub("\s+", " ", text)
    text = text.lower()
    return text

def search_engine(vectorizer, data_tfi, dados, title="Dr Seuss American icon"):
    dados_copiados = dados.copy() 
    title = title.lower()
    title = re.sub("\s+", " ", title)
    title = re.sub("[^a-zA-Z0-9\s]", "", title)
    title = title.strip(" ")
    vetor_titulo = vectorizer.transform([title])
    similarities_title = cosine_similarity(vetor_titulo, data_tfi).flatten()
    dados_copiados["similarity_title"] = similarities_title
    resultado = dados_copiados.sort_values(by="similarity_title", ascending=False).head(3)
    return resultado


def similaridade(vetor_dataframe, vetor_amostra):
    similaridades = cosine_similarity(torch.cat(vetor_dataframe.tolist()), vetor_amostra.numpy().reshape(1, -1))
    return similaridades


st.markdown("# Book Advisor :book:")
st.subheader('I would like to suggest you a new book!!')
data = loading_data(r"data\preprocessed_data.parquet")
embendding_matrix = loading_embedding_bert(r'data\vetor_embedding_bert_large.pkl')
title_input = limpando_titulos(st.sidebar.text_input(label="Write a Title", value="Dr Seuss American icon"))

tfid_model = joblib.load(r"models\tfid.joblib")
vetor_transformado = tfid_model.transform(data["Title_cleaned"])
titulo_escolhido = search_engine(tfid_model, vetor_transformado, data, title_input)

condicao_nao_existencia = titulo_escolhido.values[0, -1] > 0.7
if not titulo_escolhido["Title"].empty and condicao_nao_existencia:
    with st.sidebar:
            titulo1 = titulo_escolhido.reset_index(drop=True).loc[[0]]
            titulo2 = titulo_escolhido.reset_index(drop=True).loc[[1]]
            titulo3 = titulo_escolhido.reset_index(drop=True).loc[[2]]
            st.subheader('Are any of these your book?')
            plotar_dados(titulo1)
            button1 = st.button(label="First option")
            plotar_dados(titulo2)
            button2 = st.button(label="Second option")
            plotar_dados(titulo3)
            button3 = st.button(label="Third option")
            if button1:
                titulo_escolhido = titulo1
            elif button2:
                titulo_escolhido = titulo2
            else:
                titulo_escolhido = titulo3
else:
    with st.sidebar:
        st.write("### I do not have this book in my database!")

left_column, middle_column, right_column = st.columns(3, gap="large")
if condicao_nao_existencia and (button1 or button2 or button3):
    index = titulo_escolhido.index[0]
    vetor_embebending_amostra = embendding_matrix.iloc[index][0]

    # Filtrar os livros de categorias semelhantes:
    data_mesma_categoria = filtro_categorias(titulo_escolhido, data)
    similaridades_df = similaridade(embendding_matrix, vetor_embebending_amostra)
    df_com_similares = data.copy()
    df_com_similares["similaridade"] = similaridades_df

    # Similares
    top_gender = df_com_similares.loc[data_mesma_categoria.index].sort_values(by="similaridade", ascending=False).reset_index(drop=True).loc[1:]
    similarity_top = df_com_similares[~df_com_similares["Title"].isin(top_gender.head(4)["Title"].tolist())].sort_values(by="similaridade", ascending=False)
    similarity_top_diferentes = similarity_top.reset_index(drop=True)
    
    if len(top_gender) > 0:
        with left_column:
            primeiro_livro = top_gender.loc[[1]]
            plotar_dados(primeiro_livro)
        
        with middle_column:
            if len(top_gender) >= 2:
                segundo_livro = top_gender.loc[[2]]
                plotar_dados(segundo_livro)

        with right_column:
            if len(top_gender) >= 3:
                terceiro_livro = top_gender.loc[[3]]
                plotar_dados(terceiro_livro)   
    else:
        st.write("## There are no books of the same category!!")

    st.subheader("Books of mixed categories, that could be of your interest.")
    left_column_2, middle_column_2, right_column_2 = st.columns(3, gap="large")
    with left_column_2:
        primeiro_livro_similarity = similarity_top_diferentes.loc[[1]]
        plotar_dados(primeiro_livro_similarity)

    with middle_column_2:
        segundo_livro_similarity = similarity_top_diferentes.loc[[2]]
        plotar_dados(segundo_livro_similarity)
    
    with right_column_2:
        terceiro_livro_similarity = similarity_top_diferentes.loc[[3]]
        plotar_dados(terceiro_livro_similarity)


st.write(data.sample(n=30))