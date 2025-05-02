import streamlit as st
from sentence_transformers import SentenceTransformer
from Rag_milvus import pdfachunk,split_chunks,generaremben,insertarenqdra
from Llm_local import generarPages


col1, col2 = st.columns([1, 4])
with col1:
    st.image("Procuradurialogo.jpg", width=600)

with col2:
    st.markdown("""
        <div style='display: flex; align-items: center; height: 100%;'>
            <h1 style='margin: 0; text-align: center;'>ProcurIA</h1>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.title("Menú de Funciones")
generarPages()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

archivo = st.file_uploader("📂 Sube tus documentos PDF para chatear con ellos", type=["pdf", "txt"], accept_multiple_files=True)

if archivo:
    for file in archivo:
        nombre_coleccion = f"coleccion_{file.name.replace('.pdf', '').replace(' ', '_')}"
        pdf_chunks = pdfachunk(file)
        split_docs = split_chunks(pdf_chunks)
        texts = [doc.page_content for doc in split_docs]
        embeddings = generaremben(model, texts)
        insertarenqdra(embeddings, texts, nombre_coleccion)

    st.success("Archivos subidos en la base vectorial con éxito!")
