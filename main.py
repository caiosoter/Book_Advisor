import streamlit as st
import pandas as pd
import re
import joblib
import scipy.sparse as ss
import numpy as np
import boto3
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.parquet as pq

@st.cache_data
def loading_book_map(s3):
    #data = pq.read_table(r"data\book_id_map.parquet").to_pandas()
    obj = s3.Bucket(st.secrets["bucket_name"]).Object("book_id_map.parquet").get()["Body"]
    data =  pq.read_table(obj).to_pandas()
    return data


@st.cache_data
def loading_books(s3):
    #data = pq.read_table(r"data\goodreads_books.parquet").to_pandas()
    obj = s3.Bucket(st.secrets["bucket_name"]).Object("goodreads_books.parquet").get()["Body"]
    data = pq.read_table(obj).to_pandas()
    return data


@st.cache_data
def loading_book_interactions(s3):
    #data = pq.read_table(r"data\goodreads_interactions.parquet").to_pandas()
    obj = s3.Bucket(st.secrets["bucket_name"]).Object("goodreads_interactions.parquet").get()["Body"]
    data = pq.read_table(obj).to_pandas()
    return data

@st.cache_data
def loading_model(s3):
    #model = joblib.load(r"models\vectorizer.joblib")
    #data_tfidf = ss.load_npz(r"data\data_tfdi.npz")
    obj1 = s3.Bucket(st.secrets["bucket_name"]).Object("vectorizer.joblib").get()["Body"]
    obj2 = s3.Bucket(st.secrets["bucket_name"]).Object("data_tfdi.npz").get()["Body"]
    
    return obj1, obj2


def connection_s3():
    try:
        s3 = boto3.resource(service_name=st.secrets["service_name"],
                        region_name=st.secrets["region_name"],
                        aws_access_key_id=st.secrets["aws_access_key_id"],
                        aws_secret_access_key=st.secrets["aws_secret_access_key"])
    except ConnectionError as error:
        st.write("Connection to AWS S3 failed")
        st.write(error)
    return s3



def search_engine(title, books, tf_data, model):
    title = re.sub("[^a-zA-Z0-9 ]", "", title.lower())
    title_transformed = model.transform([title])
    similarity = cosine_similarity(title_transformed, tf_data).flatten()
    index = np.argpartition(similarity, -10)[-10:]
    resuts = books.loc[index].sort_values("ratings_count", ascending=False)
    return resuts.head(5)


def recomendacao(escolha, df_books, df_book_id_map, df_interactions):
    csv_id = df_book_id_map[df_book_id_map["book_id"].isin(escolha)].loc[:, "book_id_csv"].values[0]
    usuarios = df_interactions[(df_interactions["book_id"] == csv_id)&(df_interactions["rating"] >= 4)]
    id_books_usuarios = df_interactions[df_interactions["user_id"].isin(usuarios["user_id"])]
    id_books_usuarios = id_books_usuarios[~id_books_usuarios["book_id"].isin(escolha)]
    resultado = id_books_usuarios["book_id"].value_counts(ascending=False).to_frame().reset_index()
    resultado = pd.merge(resultado, df_book_id_map, how="inner", left_on="book_id", right_on="book_id_csv").drop(columns=["book_id_x", "book_id_csv"])
    resultado = pd.merge(resultado, df_books, how="inner", left_on="book_id_y", right_on="book_id")
    resultado = resultado[~resultado["book_id"].isin(escolha)]
    resultado["score"] = resultado["count"] * (resultado["count"]/resultado["ratings_count"])
    resultado = resultado[resultado["count"] > 75].sort_values("score", ascending=False).head(6)
    return resultado

s3 = connection_s3()
df_books = loading_books(s3)
df_interactions = loading_book_interactions(s3)
df_map_id = loading_book_map(s3)
model, data_tfid = loading_model(s3)

st.write(df_books.head())
st.write(df_interactions.head())
st.write(type(model))
st.write(type(data_tfid))






"""with st.sidebar:
    input_title = st.text_input(label="Write a title")
    
output_titles = search_engine(input_title, df_books, data_tfid, model)
st.dataframe(output_titles, column_config={"url": st.column_config.LinkColumn("Book URL")})

rec = recomendacao([output_titles["book_id"].values[0]], df_books, df_map_id, df_interactions)
st.dataframe(rec)"""
