import streamlit as st
import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer
import wikipedia
import re

# Load GPT-J model and tokenizer
@st.cache_resource
def load_gpt_j():
    model_name = "EleutherAI/gpt-j-6B"  # GPT-J 6B model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle padding token
    model = GPTJForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_j()

# Function to fetch Wikipedia data
def get_wikipedia_info(query):
    try:
        # Fetch the summary of the query from Wikipedia
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # In case of ambiguity, return a list of possible options
        return f"There are multiple topics for '{query}'. You can be more specific."

# Function to generate AI response based on user input
def generate_text(user_input, max_length=150, fetch_wiki=False):
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history]
    )
    
    if fetch_wiki:
        # If the user query is asking for factual data, get Wikipedia info
        wiki_info = get_wikipedia_info(user_input)
        prompt = f"{conversation_context}\nUser: {user_input}\nAI (with Wikipedia info): {wiki_info}\nAI:"
    else:
        # Otherwise, generate the response based on context alone
        prompt = f"{conversation_context}\nUser: {user_input}\nAI:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate output
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),  # Adjust max length
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetitive n-grams
        temperature=0.7,  # Creativity control
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AI response (after "AI:")
    response_start = generated_text.find("AI:") + len("AI:")
    return generated_text[response_start:].strip()

# Function to generate Python code from a query
def generate_python_code(query):
    prompt = f"Write a Python script for the following task:\n{query}\nPython code:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate code output
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetitive n-grams
        temperature=0.7,  # Creativity control
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# Function to improve or refactor Python code
def improve_python_code(given_code):
    prompt = f"Improve the following Python code:\n{given_code}\nImproved Python code:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate refactored code output
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetitive n-grams
        temperature=0.7,  # Creativity control
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    improved_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return improved_code

# Streamlit Interface
st.title("Advanced GPT-J AI Chatbot with Python Code Generation")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Your message:", placeholder="Ask me anything...")

# Option to enable Wikipedia-based response or code generation
fetch_wiki_info = st.checkbox("Fetch Wikipedia info (for factual queries)")
generate_code = st.checkbox("Generate Python Code (for code queries)")
improve_code = st.checkbox("Improve Given Python Code (paste code below)")

# If user asks for Python code improvement
given_code = st.text_area("Paste code to improve:", height=200)

if user_input:
    with st.spinner("Generating response..."):
        if generate_code:
            # Generate Python code based on the user's query
            ai_response = generate_python_code(user_input)
        elif improve_code and given_code:
            # Improve or refactor the given code
            ai_response = improve_python_code(given_code)
        else:
            # Regular response with or without Wikipedia info
            ai_response = generate_text(user_input, fetch_wiki=fetch_wiki_info)

        st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"<div class='chat-container user-message'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-container ai-message'><strong>AI:</strong> {chat['ai']}</div>", unsafe_allow_html=True)

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.history = []
