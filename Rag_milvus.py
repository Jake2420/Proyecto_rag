from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
import numpy as np
import streamlit as st

def pdfachunk(file, chunk_size_pages=20):
    # Usar el buffer binario del archivo subido
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for i in range(0, len(doc), chunk_size_pages):
        text = ""
        for page_num in range(i, min(i + chunk_size_pages, len(doc))):
            text += doc[page_num].get_text()
        chunks.append(text)
    doc.close()
    return chunks

def split_chunks(raw_chunks, chunk_size=1024, chunk_overlap=100):
    docs = [Document(page_content=chunk) for chunk in raw_chunks]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)

def generaremben(model, texts):
    texts = [t for t in texts if t.strip()]  # filtra vacíos
    if not texts:
        raise ValueError("No hay textos válidos para generar embeddings.")
    return model.encode(texts, batch_size=16, show_progress_bar=True)

def insertarenqdra(embeddings, texts, nombre_coleccion):
    client = QdrantClient(path="./data_v2")  # persistente

    dim = len(embeddings[0])
    client.recreate_collection(
        collection_name=nombre_coleccion,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": texts[i]})
        for i in range(len(embeddings))
    ]

    client.upsert(collection_name=nombre_coleccion, points=points)
    print(f"✅ Insertados {len(points)} vectores en Qdrant.")

def query_qdrant(query, model, nombre_coleccion, top_k, umbral):
    query_embedding = generaremben(model, [query])[0]  
    

    query_embedding = np.array(query_embedding).tolist()

    client = QdrantClient(path="./data_v2")  

    results = client.query_points(
        collection_name=nombre_coleccion,
        query=query_embedding,  
        limit=top_k,  
        with_payload=True,
        score_threshold=umbral
    )
    
    return results

def query_qdrant_sinumbral(query, model, nombre_coleccion, top_k=5):
    query_embedding = generaremben(model, [query])[0]  
    

    query_embedding = np.array(query_embedding).tolist()

    client = QdrantClient(path="./data_v2")  

    results = client.query_points(
        collection_name=nombre_coleccion,
        query=query_embedding,  
        limit=top_k,  
        with_payload=True,
    )

    return results


def obtener_colecciones(path="./data_v2"):
    client = QdrantClient(path=path)
    collections = [col.name for col in client.get_collections().collections]
    return ["Todas las colecciones"] + collections




