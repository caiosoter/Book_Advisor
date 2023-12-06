import streamlit as st
"""import mysql.connector
conn = mysql.connector.connect(
    host=st.secrets["host"],
    username=st.secrets["username"],
    password=st.secrets["password"],
    database=st.secrets["database"]
)"""

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Perform query.
df = conn.query('SELECT * from mytable;', ttl=600)

# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")