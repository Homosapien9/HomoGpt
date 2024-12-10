import streamlit as st
import PyPDF2
import docx
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to generate text using GPT-Neo
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit Web Interface
st.title("AI Summarizer and Chatbot")
st.write("Upload a document or chat with AI.")

# File uploader for summarization
uploaded_file = st.file_uploader("Upload a PDF or DOCX file for summarization", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text from the uploaded file
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    
    st.write("Extracted Text:")
    st.write(text[:1000])  # Display the first 1000 characters for preview

    # Summarize the extracted text
    if st.button("Summarize Document"):
        if text:
            with st.spinner("Summarizing..."):
                summary = generate_text(text, max_length=200)
                st.write("Summary:")
                st.write(summary)
        else:
            st.error("No content to summarize.")

# Chatbot functionality
st.header("Chat with AI")
user_input = st.text_input("Enter your message:")

if user_input:
    with st.spinner("Generating response..."):
        response = generate_text(user_input, max_length=150)
        st.write("AI Response:")
        st.write(response)
