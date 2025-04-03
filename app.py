import streamlit as st
import time
import fitz
import io
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
# import shutil
from PIL import Image
import pytesseract
# from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from chromadb.config import Settings
# ========== CONFIG ==========
st.set_page_config(page_title="PDF Q&A Generator", layout="wide")

# ========== STREAMLIT HEADER ==========
st.title("PDF Q&A Generator")

# ========== FILE UPLOAD ==========
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Only process if a file is uploaded
if uploaded_file is not None:
    start_time = time.time()
    st.success("Uploaded successfully!")
    
    # ========== PROCESS PDF ==========
    with st.spinner("Extracting text from PDF..."):
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"

    with st.spinner("Applying OCR on images..."):
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image = Image.open(io.BytesIO(base_image["image"]))
                text += pytesseract.image_to_string(image) + "\n"

    with st.spinner("Splitting text into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

    # ========== VECTOR DATABASE MANAGEMENT ==========
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        st.info("Using existing vector database...")
        vector_db = Chroma(persist_directory=db_path)
    else:
        with st.spinner("Generating new embeddings..."):
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            chroma_settings = Settings(persist_directory=db_path, anonymized_telemetry=False)
            vector_db = Chroma.from_documents(documents, embedding_model, persist_directory=db_path, client_settings=chroma_settings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # ========== GENERATE QUESTIONS ==========
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
        
        # Simple error handling for API key
    if not api_key:
            st.error("GROQ API Key is not read")
            st.stop()
            
    llm = ChatGroq(model_name="gemma2-9b-it", api_key=api_key)

    query = "Generate a set of questions and answers based only on the given PDF content."
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
            Generate a comprehensive set of high-quality questions and answers based ONLY on the provided text. **Maximize** the number of unique questions while ensuring technical depth.

            ### INSTRUCTIONS:
            1. **Generate at least 60 questions**, evenly distributed across three difficulty levels:
            - **Easy:** Factual recall and direct technical questions (15-20)
            - **Medium:** Conceptual understanding and application-based questions (15-20)
            - **Hard:** Deep analysis, real-world application, and critical thinking (15-20)

            ---
            ### **FORMAT REQUIREMENTS**
            - **Strict formatting:**
            - ***Strictly follow markdown format* for generating questions and answers.
            - **Use a structured Markdown layout** for easy reading and display.
            - **Questions:** Bolded and numbered (e.g., Q1.)
            - **Answers:** Bolded with the letter 'A' (e.g., A.)
            - **Code Snippets:** Use triple backticks (\`) for code blocks.
            - **Divide questions by difficulty levels** (Easy, Medium, Hard).
            - Give the difficulty level titles a bold and bigger text.
            - **Questions and answers must be in consecutive lines, and there must be a blank line between two questions.**

            # Easy Questions
            ---

            <details>
            <summary><strong>Q1.</strong> [Question here]</summary>
            <p><strong> Answer:</strong> Answer here</p>
            </details>

            ---

            <details>
            <summary><strong>Q2.</strong> [Question here]</summary>
            <p><strong> Answer:</strong> Answer here</p>
            </details>

            ---

            # Medium Questions
            ---

            <details>
            <summary><strong>Q1.</strong> [Question here]</summary>
            <p><strong> Answer:</strong> Answer here</p>
            </details>

            ---

            # Hard Questions
            ---

            <details>
            <summary><strong>Q1.</strong> [Question here]</summary>
            <p><strong> Answer:</strong> Answer here</p>

            ```python
            # Example Code Snippet
            print("Hello, World!")

            3. **CRITICAL RULES - MUST FOLLOW:**
            - Use **only** the substantive content from the text.
            - **Ensured** code snippets are readable and correctly formatted. 
            - **Exclude** metadata, author details, document creation info, external references, or document title.
            - Never reference the document itself in any way.
            - **Avoid** yes/no questions or questions with one-word answers.
            - **Ensure** that questions are **unique** and **non-redundant**.
            - **Avoid** questions that are too specific or too general.
            - **Include** a variety of question types (e.g., multiple-choice, fill-in-the-blank, scenario-based).
            - **Ensure** that the questions are **clear, concise, and grammatically correct**.
            - **Include** a mix of questions that test **factual recall, conceptual understanding, and critical thinking**.
            - **Avoid** questions that are too easy or too difficult.
            - **Ensure** that the answers are **accurate, complete, and well-explained**.

            4. **QUESTION GENERATION STRATEGY:**
            - Extract every possible fact, number, date, term, or concept.
            - Frame multiple questions from each paragraph covering different angles.
            - Test relationships between concepts.
            - Create **scenario-based** and **"What if"** questions.
            - Include explanations of processes, hierarchies, and systems.
            - In **Easy**, emphasize technical accuracy and definitions.
            - In **Medium**, assess conceptual clarity with moderate application.
            - In **Hard**, emphasize **practical application** and **deep understanding**.
            - In **Hard** questions, if the given text contains any programming concepts, include code snippet questions where the user must read and understand the code to answer the question. Give the code snippet surrounded by ``. Stick to the language in the text.

            5. **ADDITIONAL REQUIREMENTS:**
            - Avoid redundant or repetitive questions.
            - Ensure concise yet complete answers.
            - **Prioritize technical questions** in **Easy** while keeping half of the **Medium**'s technical.
            - **For Hard questions, focus on in-depth application and reasoning.**

            ### TEXT:
            {context}
            """

    
    # === Get Q&A from LLM ===
    with st.spinner("Generating Q&A from LLM..."):
        qa_res = llm.invoke(prompt)
        qa_text = qa_res.content

    end_time = time.time()
    
    st.markdown(qa_text, unsafe_allow_html=True)
    st.success(f"Processing completed in {start_time - end_time} seconds!")
else:
    st.info("Upload a PDF to generate questions and answers.")