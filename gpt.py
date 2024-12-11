import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# Cached Resources for Efficient Loading
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model, tokenizer

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize Models
gpt_neo_model, gpt_neo_tokenizer = load_gpt_neo()
summarizer = load_summarizer()

# Text Generation Function
def generate_text(prompt, max_length=150):
    inputs = gpt_neo_tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256  # Reduce input size to save memory
    ).to(gpt_neo_model.device)

    outputs = gpt_neo_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + inputs["input_ids"].shape[1],
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=30,
        top_p=0.8,
        pad_token_id=gpt_neo_tokenizer.pad_token_id
    )

    return gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# Summarization Function
def summarize_text(input_text, max_length=150):
    return summarizer(input_text[:1000], max_length=max_length, min_length=50, do_sample=False)[0]["summary_text"]

# Web Search Function (DuckDuckGo)
def live_web_search(query, max_results=3):
    search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.find_all("a", class_="result__a")[:max_results]:
        title = result.text
        link = result["href"]
        snippet = result.find_next("a", class_="result__snippet")
        snippet_text = snippet.text if snippet else "No snippet available."
        results.append({"title": title, "link": link, "snippet": snippet_text})
    return results

# Python Code Generation Function
def generate_python_code(prompt):
    code_prompt = f"Write Python code for the following task:\n{prompt}"
    return generate_text(code_prompt, max_length=300)

# Streamlit Interface
st.title("Advanced AI Chatbot")
st.markdown(
    """
    **Features:**
    - Real-Time Web Search using DuckDuckGo
    - AI-Powered Text Generation and Summarization
    - Python Code Generation
    """
)

# Session State Initialization
if "history" not in st.session_state:
    st.session_state["history"] = []

# User Input
user_input = st.text_area("Your question, command, or Python code task:")
if user_input:
    with st.spinner("Processing..."):
        if "search" in user_input.lower():
            st.subheader("Search Results")
            for result in live_web_search(user_input):
                st.markdown(f"**[{result['title']}]({result['link']})**")
                st.write(result["snippet"])
        elif "code" in user_input.lower():
            st.subheader("Generated Python Code")
            python_code = generate_python_code(user_input)
            st.code(python_code, language="python")
        else:
            response = generate_text(user_input)
            st.session_state["history"].append(f"User: {user_input}\nAI: {response}")
            st.write(response)

# Conversation History
if st.session_state["history"]:
    st.subheader("Conversation History")
    for entry in st.session_state["history"]:
        st.markdown(entry)

# Clear Chat
if st.button("Clear Chat"):
    st.session_state["history"] = []
