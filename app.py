import streamlit as st
import google.generativeai as genai
import PyPDF2
import time
from google.api_core import retry
from google.api_core import exceptions  # Import the exceptions module

# Set up the Gemini API key
genai.configure(api_key="gemini api key") ## use your own api key here, here is the line to generate api key "https://ai.google.dev/gemini-api/docs/api-key"

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file, password=None):
    reader = PyPDF2.PdfReader(uploaded_file)
    if reader.is_encrypted:
        if password:
            reader.decrypt(password)
        else:
            raise ValueError("The PDF is encrypted. Please provide a password.")
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, max_tokens=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_tokens:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize the text using Gemini AI
@retry.Retry(
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=60.0,
    predicate=retry.if_exception_type(
        exceptions.ResourceExhausted,  # Use the imported exceptions module
    ),
)
def summarize_text(text):
    model = genai.GenerativeModel('gemini-pro')
    chunks = split_text(text)
    summaries = []
    for chunk in chunks:
        response = model.generate_content(f"Summarize the following research paper:\n{chunk}")
        summaries.append(response.text)
        time.sleep(1)  # Add a delay between requests
    return " ".join(summaries)

# Function to answer user questions using Gemini AI
def answer_question(text, question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Based on the following research paper:\n{text}\n\nAnswer the following question:\n{question}")
    return response.text

# Streamlit app
def main():
    st.title("Research Paper Summarizer and Q&A Chatbot")
    st.write("Upload a PDF file to summarize and ask questions about it.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    #password = st.text_input("Enter PDF password (if encrypted):", type="password")

    if uploaded_file is not None:
        try:
            # Extract text from the uploaded PDF
            text = extract_text_from_pdf(uploaded_file)
            st.success("PDF file uploaded and text extracted successfully!")

            # Summarize the text
            st.subheader("Summary of the Research Paper")
            summary = summarize_text(text)
            st.write(summary)

            # Interactive Q&A
            st.subheader("Ask a Question")
            question = st.text_input("Enter your question about the research paper:")
            if question:
                answer = answer_question(text, question)
                st.write("**Answer:**")
                st.write(answer)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
