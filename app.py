import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

st.set_page_config(page_title="CV RAG Chatbot", page_icon="Resume", layout="centered")
st.title("CV / Resume Analyzer with RAG")
st.markdown("*Upload a PDF resume and ask any question — powered by Groq + Llama-3.1*")


with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get it free at https://console.groq.com/keys")
    st.info("Your key is never stored — it's only used during this session.")
    
    model_choice = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "filename" not in st.session_state:
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload CV / Resume (PDF)", type="pdf")

if uploaded_file and groq_api_key:
    if st.session_state.filename != uploaded_file.name:
        with st.spinner(f"Processing {uploaded_file.name}... This may take 10–30 seconds."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load and split
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Embeddings & Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.filename = uploaded_file.name

            os.unlink(tmp_path)  # Clean up

        st.success(f"Loaded & indexed: **{uploaded_file.name}** ({len(splits)} chunks)")


def get_rag_chain():
    if not st.session_state.vectorstore:
        return None

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_choice,
        temperature=temperature
    )

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are an expert HR and Recruitment Assistant analyzing a candidate's CV/Resume. "
        "Use only the retrieved context to answer. "
        "If the information is not present, say: 'I cannot find that information in the provided CV.' "
        "Be concise, professional, and structured."
        "\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if st.session_state.vectorstore and groq_api_key:
    st.markdown(f"### Chatting with: **{st.session_state.filename}**")

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # User input
    if prompt := st.chat_input("Ask something about the CV (e.g., work experience, skills, education)..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_rag_chain()
                response = chain.invoke({"input": prompt})
                answer = response["answer"]

                st.write(answer)

                # Optional: Show sources
                # with st.expander("View retrieved sources"):
                #     for i, doc in enumerate(response.get("context", []), 1):
                #         st.markdown(f"**Source {i}:**")
                #         st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                #         st.caption(f"*Page {doc.metadata.get('page', 'Unknown')}*")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
    if not uploaded_file:
        st.info("Upload a PDF resume to get started!")

# ============================
# Footer
# ============================
st.markdown("---")
st.caption("Built by Muhammad Ali Qamar | Using Groq • LangChain • FAISS • all-MiniLM-L6-v2")