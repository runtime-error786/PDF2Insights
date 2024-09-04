
import streamlit as st
import fitz
import os
import PyPDF2
import base64
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, DocumentCompressorPipeline
from langchain_cohere import CohereRerank
from langchain.chains.combine_documents import create_stuff_documents_chain

if 'memory' not in st.session_state:
    st.session_state.memory = []
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'full_text' not in st.session_state:
    st.session_state.full_text = None
if 'embedding_db' not in st.session_state:
    st.session_state.embedding_db = None
if 'history_aware_retriever' not in st.session_state:
    st.session_state.history_aware_retriever = None
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None
result = ''
store = {}

booli = True

def convert_pdf_to_images(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = f"{pdf_name}_images"
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    saved_image_paths = []
    
    for i, page in enumerate(pdf_document):
        pix = page.get_pixmap()
        image_path = os.path.join(output_dir, f'page{i}.jpg')
        pix.save(image_path)
        saved_image_paths.append(image_path)
    
    return saved_image_paths

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Failed to extract text from PDF. Error: {e}")
    return text

def process_image_and_return_text(image_path):
    with open(image_path, 'rb') as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('ascii')
    llm = Ollama(model="llava:13b")
    prompt = """
You are an expert in RAG (Retrieval-Augmented Generation) and NLP tasks. Analyze the following image and provide a concise explanation based on the content:

**For Tables:**
- Describe the table and its purpose.
- List and briefly explain each column header.
- Summarize the data in each row.

**For Charts or Graphs:**
- Identify the chart type and describe the axes and legends.
- Provide a brief analysis of the data trends and key insights.

**For Diagrams:**
- Describe the diagram’s layout and components.
- Summarize the process or workflow depicted.

**General Instructions:**
- Keep the explanation clear and in detail view analysis of each things i mentioned above.
- Focus on the most relevant information related to RAG and NLP tasks.
"""
    try:
        response = llm.generate(prompts=[prompt], images=[encoded_image])
        if response and hasattr(response, 'generations'):
            generations = response.generations
            if generations and generations[0]:
                text = generations[0][0].text if generations[0][0] else "No text available."
                summary_text = text
            else:
                summary_text = "No generations available."
        else:
            summary_text = "Unexpected response format."
    except Exception as e:
        summary_text = f"Failed to process the image. Error: {str(e)}"
    return summary_text

def aggregate_text_from_images(directory):
    if not os.path.isdir(directory):
        st.error(f"Directory '{directory}' does not exist.")
        return ""
    
    full_text = ""
    try:
        for filename in sorted(os.listdir(directory)):  
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(directory, filename)
                
                

                text = process_image_and_return_text(image_path)
                

                full_text += text + "\n\n"
                
        st.write("Text aggregation complete.")
        return full_text
    except Exception as e:
        st.error(f"Failed to aggregate text from images. Error: {e}")
        return ""

def process_text(full_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding=embedding, persist_directory="./chroma_db/embedding")
    retriever = db.as_retriever(search_kwargs={"k": 6})

    llm_text = Ollama(model="llama3")
    COHERE_API_KEY = "Q1tNApphbMywrTviu1WfdEYa3DfNr8NtwhlGAiYh"
    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY)

    _filter = LLMChainFilter.from_llm(llm_text)
    embeddings_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(transformers=[reranker, LLMChainExtractor.from_llm(llm_text)])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm_text, compression_retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an expert assistant for answering questions accurately and in detail. "
        "You must base your response strictly on the provided context. "
        "If the context does not contain enough information to answer the question, respond with 'I don't know.' "
        "Avoid using any information not present in the context but use related information from your knowledge. "
        "Ensure your answer is clear and directly addresses the question but give answer in detail.\n\n"
        "Context:\n{context}\n"
        "Question: {input}\n"
        "Answer:"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm_text, qa_prompt)
    print(question_answer_chain)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    st.session_state.embedding_db = db
    st.session_state.history_aware_retriever = history_aware_retriever
    st.session_state.conversational_rag_chain = conversational_rag_chain

st.title("PDF to Text and Image Processing")

pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
if pdf_file:
    if st.session_state.full_text is None:  
        pdf_path = os.path.join("data", pdf_file.name)
        text_file_path = "extracted/extracted_text.txt"

        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        extracted_text = extract_text_from_pdf(pdf_path)

        image_paths = convert_pdf_to_images(pdf_path)
        image_dir = os.path.dirname(image_paths[0])

        text = aggregate_text_from_images(image_dir)
        full_text = extracted_text + "\n" + text
        st.session_state.full_text = full_text

        process_text(full_text)


user_question = st.text_input("Ask a question related to the PDF content:")

if st.button("Get Answer"):
    if user_question:
        try:
            st.session_state.memory.append({"role": "user", "content": user_question})
            
            answer = st.session_state.conversational_rag_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": "abc21"}}
            )["answer"]
            st.session_state.memory.append({"role": "ai", "content": answer})

            # Display chat history
            for message in st.session_state.memory:
                role = "User" if message["role"] == "user" else "AI"
                st.write(f"**{role}:** {message['content']}")
                
        except Exception as e:
            st.error(f"Failed to get answer. Error: {e}")
    else:
        st.error("Please enter a question.")
