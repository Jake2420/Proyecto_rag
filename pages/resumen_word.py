import streamlit as st
from sentence_transformers import SentenceTransformer
from Llm_local import generarPages, informes_mistral,extraer_texto, extraer_texto_word, generar_docx, generar_pdf

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

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola!, en qué puedo ayudarte?"}]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

archivo = st.file_uploader("📂 Sube tus documentos PDF o WORD ¡Para hacer un informe!", type=["pdf", "txt"], accept_multiple_files=True)

if archivo:
    texto_total = ""

    for archivo_item in archivo:
        contenido = ""

        if archivo_item.name.endswith(".pdf"):
            contenido = extraer_texto(archivo_item)
        elif archivo_item.name.endswith(".docx"):
            contenido = extraer_texto_word(archivo_item)

        if contenido.strip():
            texto_total += f"\n---\nDOCUMENTO: {archivo_item.name}\n{contenido.strip()}\n"

    if texto_total.strip():
        resumen_completo = [""]  

        def resumen_streaming():
            for palabra in informes_distilgpt2(texto_total):
                resumen_completo[0] += palabra  
                yield palabra

        st.subheader("📄 Informe de todos los documentos:")
        st.write_stream(resumen_streaming())

        # Descargar como PDF
        pdf_bytes = generar_pdf(resumen_completo[0])
        st.download_button("📄 Descargar Informe en PDF", data=pdf_bytes, file_name="resumen_global.pdf", mime="application/pdf")

        # Descargar como Word
        docx_buffer = generar_docx(resumen_completo[0])
        st.download_button("📝 Descargar Informe en Word", data=docx_buffer, file_name="resumen_global.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
