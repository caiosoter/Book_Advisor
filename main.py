import streamlit as st
import pandas as pd
import re
import joblib
from io import BytesIO
import tempfile
import scipy.sparse as ss
import numpy as np
import os
import boto3
import logging
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Book Advisor", page_icon=":book")


@st.cache_resource
def get_s3_client():
   s3 = boto3.client('s3', 
         aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
         aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
   return s3


@st.cache_resource
def load_model_from_s3(bucket, key):
    s3_client = get_s3_client()
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3_client.download_fileobj(Fileobj=temp_file, Bucket=bucket, Key=key)
            temp_file.seek(0)
            model = joblib.load(temp_file)
        return model
    except Exception as e:
        logging.exception(e)
    finally:
        # Ensure the temporary file is deleted
        if 'temp_file' in locals():
            os.unlink(temp_file.name)


    
@st.cache_resource
def loading_tfdi():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="data_tfdi_reduzido.npz")["Body"].read()
    dados = ss.load_npz(BytesIO(obj))
    return dados

@st.cache_resource
def loading_tfdi_author():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="data_tfdi_autores_reduzido.npz")["Body"].read()
    dados = ss.load_npz(BytesIO(obj))
    return dados


@st.cache_data(max_entries=1)
def loading_books():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="goodreads_books_reduzido.feather")["Body"].read()
    dados = pd.read_feather(BytesIO(obj))
    return dados


@st.cache_data(max_entries=1)
def loading_interactions():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key=f"interactions/goodreads_interactions_reduzido.parquet")["Body"].read()
    dados = pd.read_parquet(BytesIO(obj))
    return dados

    
def plotar_dados(df):
        author = df["author"].tolist()[0]
        nome = df["Title"].tolist()[0]
        url = df["url"].tolist()[0]
        imagem = df["image"].tolist()[0]
        st.image(imagem)
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**More information**: {url}")



def search_engine_authors(author, df_books, tf_data, model):
    book_copy = df_books.copy()
    author = re.sub("[^a-zA-Z0-9 ]", "", author.lower())
    author_transformed = model.transform([author])
    similarity = cosine_similarity(author_transformed, tf_data).flatten()
    book_copy["similarites_total"] = similarity * book_copy["similarities"]
    results = book_copy.sort_values("similarites_total", ascending=False).head(5)
    return results


def search_engine(title, books, tf_data, model):
    book_copy = books.copy()
    title = re.sub("[^a-zA-Z0-9 ]", "", title.lower())
    title_transformed = model.transform([title])
    similarity = cosine_similarity(title_transformed, tf_data).flatten()
    book_copy["similarities"] = similarity
    return book_copy



def recomendacao(df_interactions, escolha, df_books):
    csv_id = df_books[df_books["book_id"].isin(escolha)]["book_id_csv"].values
    usuarios = df_interactions[(df_interactions["book_id"].isin(csv_id))&(df_interactions["rating"] >= 4)]
    id_books_usuarios = df_interactions[df_interactions["user_id"].isin(usuarios["user_id"])]
    id_books_usuarios = id_books_usuarios[~id_books_usuarios["book_id"].isin(csv_id)]
    resultado = id_books_usuarios["book_id"].value_counts(ascending=False).to_frame().reset_index()
    resultado = pd.merge(resultado, df_books, how="inner", left_on="book_id", right_on="book_id_csv")
    resultado["score"] = resultado["count"] * (resultado["count"]/resultado["ratings_count"])
    resultado = resultado.drop(columns=["book_id_x"]).rename(columns={"book_id_y":"book_id"})
    resultado = resultado.sort_values(["score" , "count"], ascending=[False, False]).head(6)
    return resultado


st.markdown("# Book Advisor :book:")
st.subheader('I would like to suggest you a new book!!')
df_books = loading_books()
model = load_model_from_s3("databook", "vectorizer_reduzido.joblib")
model_autor = load_model_from_s3("databook", "vectorizer_autores_reduzido.joblib")
dados_npz = loading_tfdi()
tfd_autor = loading_tfdi_author()

with st.sidebar:
    st.subheader("Choose three titles:")
    with st.container(border=True):
        input_title = st.text_input(label="Write a title", value="Blood Oranges")
        author1 = st.text_input(label="First author", value="Anne O'Gleadra")

    with st.container(border=True):
        input_title2 = st.text_input(label="Write a second title", value="Darklight")
        author2 = st.text_input(label="Second author", value="Chad Kultgen")

    with st.container(border=True):
        input_title3 = st.text_input(label="Write a third title", value="A Spark of Heavenly Fire")
        author3 = st.text_input(label="Third author", value="Pat Bertram")

    button_response = st.button(label=":blue[Click to run]", use_container_width=True)
    if button_response:
        resultado = search_engine(input_title, df_books, dados_npz, model)
        resultado2 = search_engine(input_title2, df_books, dados_npz, model)
        resultado3 = search_engine(input_title3, df_books, dados_npz, model)

        # Filtro de autor:
        resultado = search_engine_authors(author1, resultado, tfd_autor, model_autor)
        resultado2 = search_engine_authors(author2, resultado2, tfd_autor, model_autor)
        resultado3 = search_engine_authors(author3, resultado3, tfd_autor, model_autor)

        existencia1 = resultado["similarites_total"].max() > 0.5
        existencia2 = resultado2["similarites_total"].max() > 0.5
        existencia3 = resultado3["similarites_total"].max() > 0.5

    
    st.write("**By Caio Sóter**")
    st.markdown("[![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/caiosoter/Book_Advisor)")
    st.markdown("[![](https://databook.s3.us-east-2.amazonaws.com/icons8-linkedin-48.png)](https://www.linkedin.com/in/caio-soter/)")
    
    
if button_response and (input_title and input_title2 and input_title3) and (existencia1 and existencia2 and existencia3):
    st.write("## About your books:")
    left, middle, right = st.columns(3, gap="large")
    with left:
        plotar_dados(resultado.iloc[[0]])
    with middle:
        plotar_dados(resultado2.iloc[[0]])
    with right:
        plotar_dados(resultado3.iloc[[0]])

    id_escolhido1 = resultado.iloc[[0]]["book_id"].values[0]
    id_escolhido2 = resultado2.iloc[[0]]["book_id"].values[0]
    id_escolhido3 = resultado3.iloc[[0]]["book_id"].values[0]

    df_interactions = loading_interactions()
    rec = recomendacao(df_interactions, [id_escolhido1, id_escolhido2, id_escolhido3], df_books)

    if not rec.empty:
        st.write("## My recommendations are:")
        left, middle, right = st.columns(3, gap="large")
        left_2, middle_2, right_2 = st.columns(3, gap="large")   
        with left:
            plotar_dados(rec.iloc[[0]])
        with middle:
            plotar_dados(rec.iloc[[1]])
        with right:
            plotar_dados(rec.iloc[[2]])
        with left_2:
            plotar_dados(rec.iloc[[3]])
        with middle_2:
            plotar_dados(rec.iloc[[4]])
        with right_2:
            plotar_dados(rec.iloc[[5]])

elif button_response and ((input_title and author1) and (input_title2 and author2) and (input_title3 and author3)) and (not existencia1 or not existencia2 or not existencia3):
    st.write("## Sorry, I do not have this book in my Dataset!!")
    dicionario = {input_title:[existencia1, author1], input_title2:[existencia2, author2], input_title3:[existencia3, author3]}
    for chave, value in dicionario.items():
        if not value[0] or not value[1]:
            st.write(f"##### - {chave} from the author {value[1]}")
            
elif button_response and ((input_title and not author1) or (input_title2 and not author2) or (input_title3 and not author3)):
    st.write("**Write all the three author's name, please!**")
    
elif button_response and ((not input_title and author1) or (not input_title2 and not author2) or (not input_title3 and not author3)):
    st.write("**Write all the three titles please!**")
    
elif button_response:
    st.write("## Feel free to write somes books!!")
    
else:
    st.write("##### Please click on the button if you want the recommendations.")
    


