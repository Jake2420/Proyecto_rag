import  streamlit as st 
import time
from Rag_milvus import query_qdrant, obtener_colecciones, query_qdrant_sinumbral
from Llm_local import generarPages, get_response_from_distilgpt2
from sentence_transformers import SentenceTransformer

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
#Inicializamos nuestro historial de chat
if "messages" not in st.session_state:
     st.session_state.messages = [{"role": "assistant", "content": "Hola!, en que puedo ayudarte?"}]

#Definimos modelo
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#Elegimos una coleccion
colecciones = obtener_colecciones()
coleccion_seleccionada = st.sidebar.selectbox("Selecciona una colección", colecciones)

# Mostrar el historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tus dudas"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if coleccion_seleccionada == "Todas las colecciones":
            colecciones_disponibles = obtener_colecciones()
            results = []
            umbral=1
            for coleccion in colecciones_disponibles[1:]:  
                coleccion_results = query_qdrant_sinumbral(prompt,model,coleccion)
                results.extend(coleccion_results) 
        else:
            umbral=0.56
            results = query_qdrant(prompt, model, coleccion_seleccionada,5,umbral)

        if not results:
            response = "Disculpa, no tengo información para responder esa pregunta."
        else:
            response = get_response_from_distilgpt2(prompt, results)
            for word in response.split():
                st.write(word, end="")  
                time.sleep(0.05)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    #st.write(results)
