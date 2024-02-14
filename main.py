import streamlit as st
import pandas as pd
import re
import joblib
from io import BytesIO
import tempfile
import scipy.sparse as ss
import os
import boto3
import logging
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Book Advisor", page_icon=":book")

page_bg_img = f"""
<style>
.stApp {{
             background: url({st.secrets["link_imagem_back"]});
             background-size:110% 100%;
             background-repeat: no-repeat;
             z-index: 0;
         }}

[data-testid="stHeader"] {{
background-image: linear-gradient(45deg, black, #212021);
}}

[data-testid="stWidgetLabel"]{{
    color: white
}}

[data-testid="stSidebar"]{{
background-image: linear-gradient(45deg, black, #212021);
}}

[data-testid="stSideBarUserContent"]{{
    padding: 0rem 1.5rem;
}}

.st-emotion-cache-124fx4h {{
  width: 90%;
  margin: 14px;
  background-color: rgb(1, 1, 1);
  border: 1px solid rgba(255, 255, 255, 0.2);  
}}

.st-emotion-cache-r421ms {{
    border: 1px solid rgba(255, 255, 255, 0.2);
}}


div.st-emotion-cache-1xw8zd0.e1f1d6gn0{{
    background-color: #212021;
    opacity:0.9;
}}
div.st-emotion-cache-1fjr796.e1f1d6gn3{{
    background-color: #212021;
    opacity:0.9;
    padding: 0.5rem 1.5rem;
    border-style: solid;
    border-radius: 10px;
    border-width: 2px;
    border-color: darkgray;
}}
</style>
"""



#st-emotion-cache-1fjr796.e1f1d6gn3

st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_s3_client():
   s3 = boto3.client('s3', 
     aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
         aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
   return s3


@st.cache_resource(show_spinner=False)
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
        if 'temp_file' in locals():
            os.unlink(temp_file.name)


@st.cache_resource(show_spinner=False)
def loading_tfdi():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="data_tfdi_reduzido.npz")["Body"].read()
    dados = ss.load_npz(BytesIO(obj))
    return dados

@st.cache_resource(show_spinner=False)
def loading_tfdi_author():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="data_tfdi_autores_reduzido.npz")["Body"].read()
    dados = ss.load_npz(BytesIO(obj))
    return dados


@st.cache_data(max_entries=1, show_spinner=False)
def loading_books():
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=st.secrets["bucket_name"], Key="goodreads_books_reduzido.feather")["Body"].read()
    dados = pd.read_feather(BytesIO(obj))
    return dados


@st.cache_data(max_entries=1, show_spinner=False)
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
        st.write(f"""<p style="color:white">Name: {nome} </p>""", unsafe_allow_html=True)
        st.write(f"""<p style="color:white"> Author: {author} </p>""", unsafe_allow_html=True)
        st.write(f"""<p style="color:white"> More information: </p>""", unsafe_allow_html=True)
        st.markdown(f"""{url}""")


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


right_text, left_text = st.columns([0.3, 0.5])
with right_text:
    st.markdown("""<h1 style="color:white">Book Advisor</h1>""", unsafe_allow_html=True)
with left_text:
    st.write("# :book:")
    

bucket_name = st.secrets["bucket_name"]
df_books = loading_books()
model = load_model_from_s3(bucket_name, "vectorizer_reduzido.joblib")
model_autor = load_model_from_s3(bucket_name, "vectorizer_autores_reduzido.joblib")
dados_npz = loading_tfdi()
tfd_autor = loading_tfdi_author()

with st.sidebar:
    st.markdown("""<h2 style="color:white"> Choose three titles: <h2>""", unsafe_allow_html=True)
    with st.container(border=True):
        input_title = st.text_input(label="Write a title", value="Blood Oranges")
        resultado_total1 = search_engine(input_title, df_books, dados_npz, model)
        resultado = resultado_total1[resultado_total1["similarities"] > 0.65].sort_values("similarities", ascending=False)
        author1 = st.selectbox(label="First author", options=resultado["author"].unique(), index=None)        

    with st.container(border=True):
        input_title2 = st.text_input(label="Write a second title", value="Darklight")
        resultado_total2 = search_engine(input_title2, df_books, dados_npz, model)
        resultado2 = resultado_total2[resultado_total2["similarities"] > 0.65].sort_values("similarities", ascending=False)
        author2 = st.selectbox(label="Second author", options=resultado2["author"].unique(), index=None)

    with st.container(border=True):
        input_title3 = st.text_input(label="Write a third title", value="A Spark of Heavenly Fire")
        resultado_total3 = search_engine(input_title3, df_books, dados_npz, model)
        resultado3 = resultado_total3[resultado_total3["similarities"] > 0.65].sort_values("similarities", ascending=False)
        author3 = st.selectbox(label="Third author", options=resultado3["author"].unique(), index=None)

    button_response = st.button(label=":blue[Click to run]", use_container_width=True)
    if button_response and (author1 and author2 and author3):
        resultado_total1 = search_engine_authors(author1, resultado_total1, tfd_autor, model_autor)
        resultado_total2 = search_engine_authors(author2, resultado_total2, tfd_autor, model_autor)
        resultado_total3 = search_engine_authors(author3, resultado_total3, tfd_autor, model_autor)
        
        existencia1 = resultado_total1[resultado_total1["book_id"].isin(resultado["book_id"])]["similarites_total"].max() >= 0.75
        existencia2 = resultado_total2[resultado_total2["book_id"].isin(resultado2["book_id"])]["similarites_total"].max() >= 0.75
        existencia3 = resultado_total3[resultado_total3["book_id"].isin(resultado3["book_id"])]["similarites_total"].max() >= 0.75
        
    
    col1, col2, col3 = st.columns([0.3, 0.2, 0.2])
    with col1:
        st.markdown("""<p2 style="color:white">By Caio Sóter </p2>""", unsafe_allow_html=True)
    with col2:
        st.markdown("[![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/caiosoter/Book_Advisor)")
    with col3:
        st.markdown("[![](https://databook.s3.us-east-2.amazonaws.com/icons8-linkedin-48.png)](https://www.linkedin.com/in/caio-soter/)")

    
if button_response and (input_title and input_title2 and input_title3) and (author1 and author2 and author3) and (existencia1 and existencia2 and existencia3):
    st.markdown("""<h2 style="color:white"> About your books:<h2>""", unsafe_allow_html=True)
    left, middle, right = st.columns(3, gap="large")
    with left:
        plotar_dados(resultado_total1.iloc[[0]])
    with middle:
        plotar_dados(resultado_total2.iloc[[0]])
    with right:
        plotar_dados(resultado_total3.iloc[[0]])
    
    
    id_escolhido1 = resultado_total1.iloc[[0]]["book_id"].values[0]
    id_escolhido2 = resultado_total2.iloc[[0]]["book_id"].values[0]
    id_escolhido3 = resultado_total3.iloc[[0]]["book_id"].values[0]

    with st.spinner(text=""":blue[Loading my recommendations...]"""):
        df_interactions = loading_interactions()   
        rec = recomendacao(df_interactions, [id_escolhido1, id_escolhido2, id_escolhido3], df_books)

    if not rec.empty:
        st.markdown("""<h2 style="color:white"> My recommendations are:</h2>""", unsafe_allow_html=True)
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
    st.write("""<h2 style="color:white">  Sorry, I do not have this book in my Dataset!! </h2>""", unsafe_allow_html=True)
    dicionario = {input_title:[existencia1, author1], input_title2:[existencia2, author2], input_title3:[existencia3, author3]}
    for chave, value in dicionario.items():
        if not value[0] or not value[1]:
            st.write()
            st.write(f"""<p style="color:white"> - {chave} from the author {value[1]}</p>""", unsafe_allow_html=True)

elif button_response and ((not input_title and not author1) or (not input_title2 and not author2) or (not input_title3 and not author3)):
    st.write("""<h2 style="color:white">  Feel free to write somes books!! </h2>""", unsafe_allow_html=True)
           
elif button_response and (not author1 or not author2 or not author3):
    st.write("""<h2 style="color:white"> Write all the three author's name, please!</h2>""", unsafe_allow_html=True)
       
else:
    st.markdown("""<h2 style="color:white"> I would like to suggest you a new book!!</h2>""", unsafe_allow_html=True)
    with st.container(border=True) as container:
        st.write("""<h2 style="color:white"> Instructions:</h2>""", unsafe_allow_html=True)
        st.write("""
                <p style="color:white">Hello, please provide me with three books that you enjoyed, and 
                I will offer you some recommendations based on them. To view my recommendations, 
                begin by typing the title of the book you liked. Then, select the author's name 
                from the dropdown menu related to those titles you entered previously. 
                Once you have selected all three books and all the three authors, 
                you can proceed by clicking on the button to access my recommendations. 
                Thank you!</p>
                """, unsafe_allow_html=True)
