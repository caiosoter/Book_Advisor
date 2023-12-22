import streamlit as st
import pandas as pd
import re
import joblib
from io import BytesIO
import tempfile
import scipy.sparse as ss
import numpy as np
import boto3
import logging
from sklearn.metrics.pairwise import cosine_similarity

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
         with tempfile.TemporaryFile() as fp:
             s3_client.download_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
             fp.seek(0)
             return joblib.load(fp)
     except Exception as e:
         raise logging.exception(e)

    
@st.cache_data
def loading_tfdi():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="data_tfdi.npz")["Body"].read()
    dados = ss.load_npz(BytesIO(obj))
    return dados

@st.cache_data
def loading_books():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="goodreads_books2.feather")["Body"].read()
    dados = pd.read_feather(BytesIO(obj))
    return dados

@st.cache_data
def loading_interactions():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="goodreads_interactions.feather")["Body"].read()
    dados = pd.read_feather(BytesIO(obj))
    return dados

    
def plotar_dados(df):
        author = df["author"].tolist()[0]
        year = df["publishedDate"].tolist()[0]
        nome = df["Title"].tolist()[0]
        url = df["url"].tolist()[0]
        imagem = df["image"].tolist()[0]
        if imagem:
            st.image(imagem)
        else:
            st.image(st.secrets["link_sem_imagem"])
        st.write(f"**Name**: {nome}")
        st.write(f"**Author:** {author}")
        st.write(f"**Published date:** {year}")
        st.write(f"**More information**: {url}")

            

def search_engine(title, books, tf_data, model):
    book_copy = books.copy()
    title = re.sub("[^a-zA-Z0-9 ]", "", title.lower())
    title_transformed = model.transform([title])
    similarity = cosine_similarity(title_transformed, tf_data).flatten()
    index = np.argpartition(similarity, -10)[-10:]
    book_copy["similarites"] = similarity
    resuts = book_copy.iloc[index].sort_values("similarites", ascending=False)
    resuts_filtro = resuts[resuts["Title_cleaned"].str.contains(title)]
    if not resuts_filtro.empty:
        return resuts_filtro.head(5)
    return resuts.head(5)


def recomendacao(escolha, df_books, df_interactions):
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

st.set_page_config(page_title="Book Advisor", page_icon=":book")
st.markdown("# Book Advisor :book:")
st.subheader('I would like to suggest you a new book!!')
df_books = loading_books()
st.write(df_books.head())
"""model = load_model_from_s3("databook", "vectorizer.joblib")
dados_npz = loading_tfdi()



with st.sidebar:
    st.subheader("Choose three titles of your choice:")
    input_title = st.text_input(label="Write a title", value="I, Robot")
    input_title2 = st.text_input(label="Write a second title", value="The hunger games")
    input_title3 = st.text_input(label="Write a third title", value="Catching fire")

    resultado = search_engine(input_title, df_books, dados_npz, model)
    resultado2 = search_engine(input_title2, df_books, dados_npz, model)
    resultado3 = search_engine(input_title3, df_books, dados_npz, model)

    existencia1 = resultado["similarites"].max() > 0.5
    existencia2 = resultado2["similarites"].max() > 0.5
    existencia3 = resultado3["similarites"].max() > 0.5


if (input_title and input_title2 and input_title3) and (existencia1 and existencia2 and existencia3):
    dados_interactions = loading_interactions()
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
    rec = recomendacao([id_escolhido1, id_escolhido2, id_escolhido3], df_books, dados_interactions)

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

elif (input_title and input_title2 and input_title3) and (not existencia1 or not existencia2 or not existencia3):
    st.write("## Sorry, I do not have this book in my Dataset!!")
    dicionario = {input_title:existencia1, input_title2:existencia2, input_title3:existencia3}
    for chave, value in dicionario.items():
        if not value:
            st.write(f"##### - {chave}")
else:
    st.write("## Feel free to write somes books!!")"""

    
    



