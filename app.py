import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
import fitz
import io
from PIL import Image
import pytesseract
import streamlit as st
import time
import dotenv
#import manvith m nayak
dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.title("PDF Q&A Generator")
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a file", type="pdf")

if uploaded_file is not None:
    start_time = time.time()
    st.success("Uploaded successfully!")
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.sidebar.success("PDF Uploaded Successfully!")
    st.sidebar.info("**Extracting text and images...**")
    
    text = ""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
        
    images_text = "\n".join([
        pytesseract.image_to_string(Image.open(io.BytesIO(doc.extract_image(img[0])["image"])))
        for page in doc for img in page.get_images(full=True)
    ])
    text += images_text
    
    st.sidebar.success("**Text Extraction Complete!**")
    st.sidebar.info("**Splitting into chunks...**")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    st.sidebar.success("**Text Splitting and Embedding Complete!**")
    
    st.sidebar.info("**Generating questions**")
    
    llm = ChatGroq(model_name="gemma2-9b-it", api_key=api_key)
    query = "Generate a set of questions and answers based only on the given PDF content."
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Generate a **diverse and comprehensive** set of high-quality questions and answers **ONLY** from the provided text. Ensure the questions are **technically accurate** and **contextually relevant**.

    ### INSTRUCTIONS:
    1. **Generate at least 60 questions**, distributed across three categories:
    - **1 Mark Questions:** Simple factual recall and direct technical questions (15-20).
    - **2 Mark Questions:** Conceptual understanding, application-based, and intermediate-level questions (15-20).
    - **4 Mark Questions:** Deep analysis, real-world application, critical thinking, and programming questions with code snippets in **C** (15-20).
    - If the text is too short, generate fewer questions but maintain proportional distribution.
    - **Avoid redundancy** and ensure each question is unique and meaningful.

    ---

    ### **FORMAT REQUIREMENTS** (Strictly Follow)
    - Use **Markdown format** optimized for Streamlit.
    - **Questions:** Use **bold, large font** (e.g., `### **Q1. What is X?**`).
    - **Answers:** Use **bold, prefixed with 'A:'** (e.g., `**A:** The answer here.`).
    - **Code Snippets:** Enclose in triple backticks (` ``` `) with proper syntax highlighting.
    - **Categories:** Clearly separate questions into sections (1 Mark, 2 Mark, 4 Mark).
    - Use **expandable sections** (`<details>` tag in markdown) for each question-answer pair.

    ---

    ### **Question Format**
    # **1 Mark Questions**
    ---
    <details>
    <summary>Q1. What is X?</summary>
    <p>**A:** Explanation here.</p>
    </details>

    ---

    # **2 Mark Questions**
    ---
    <details>
    <summary>Q1. Explain Y?</summary>
    <p>**A:** Detailed answer here.</p>
    </details>

    ---

    # **4 Mark Questions**
    ---
    <details>
    <summary>Q1. Analyze Z?</summary>
    <p>**A:** In-depth explanation.</p>

    ```C
    // Example Code
    printf("Hello, World!");
    ```
    </details>

    ---

    ### **Context**
    {context}
    """

    # === Get Q&A from LLM ===
    with st.spinner("Generating Q&A from LLM..."):
        qa_res = llm.invoke(prompt)
        qa_text = qa_res.content

    end_time = time.time()
    
    st.markdown(qa_text, unsafe_allow_html=True)
    st.success(f"Processing completed in {end_time - start_time} seconds!")
else:
    st.info("Upload a PDF to generate questions and answers.")
