import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import wikipedia

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

gpt_neo_model, gpt_neo_tokenizer, device = load_gpt_neo()
summarizer = load_summarizer()

# Function to generate text
def generate_text(prompt, max_length=300):
    inputs = gpt_neo_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    outputs = gpt_neo_model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=gpt_neo_tokenizer.pad_token_id
    )
    generated_text = gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

# Function to summarize text
def summarize_text(input_text, max_length=150):
    return summarizer(input_text, max_length=max_length, min_length=50, do_sample=False)[0]["summary_text"]

# Advanced PDF Handling: Extract text and metadata
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    metadata = reader.metadata
    text = ""
    toc = reader.outline if hasattr(reader, 'outline') else None
    for page in reader.pages:
        text += page.extract_text()
    return text, metadata, toc

# Live Web Search using Google Search
def live_web_search(query, max_results=3):
    search_url = f"https://www.google.com/search?q={quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.find_all("div", class_="tF2Cxc")[:max_results]:
        title = g.find("h3").text
        link = g.find("a")["href"]
        snippet = g.find("span", class_="aCOpRe").text if g.find("span", class_="aCOpRe") else "No snippet available."
        results.append({"title": title, "link": link, "snippet": snippet})
    return results

# Streamlit Interface
st.title("Advanced AI Chatbot")
st.markdown("""
An enhanced AI assistant with:
- Advanced PDF handling (text, metadata, table of contents)
- Real-time web search capabilities
""")

# Initialize session states
if "history" not in st.session_state:
    st.session_state.history = []

if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting PDF content..."):
        extracted_text, pdf_metadata, pdf_toc = extract_text_from_pdf(uploaded_file)
        st.session_state.pdf_content = extracted_text
        st.subheader("PDF Metadata")
        st.write(pdf_metadata)
        if pdf_toc:
            st.subheader("Table of Contents")
            st.write(pdf_toc)
    st.success("PDF content extracted!")

# Display extracted content
if st.session_state.pdf_content:
    st.subheader("Extracted Text")
    with st.expander("View Extracted Text"):
        st.write(st.session_state.pdf_content)

    st.subheader("Summarized Text")
    with st.spinner("Summarizing extracted content..."):
        summary = summarize_text(st.session_state.pdf_content)
        st.write(summary)

# User input for queries or tasks
user_input = st.text_input("Your question or command:")

if user_input:
    with st.spinner("Processing..."):
        if "search" in user_input.lower():
            search_results = live_web_search(user_input)
            st.subheader("Search Results")
            for result in search_results:
                st.markdown(f"**[{result['title']}]({result['link']})**")
                st.write(result["snippet"])
        elif st.session_state.pdf_content:
            st.subheader("Answer Based on PDF Content")
            st.write(generate_text(f"Answer this based on the content:\n{st.session_state.pdf_content}\n{user_input}"))
        else:
            response = generate_text(user_input)
            st.session_state.history.append(f"User: {user_input}\nAI: {response}")
            st.write(response)

# Display conversation history
if st.session_state.history:
    st.subheader("Conversation History")
    for entry in st.session_state.history:
        st.markdown(entry)

# Clear chat
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.pdf_content = None
