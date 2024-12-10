import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Smaller model for faster response
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Function to generate responses based on context
def generate_text(prompt, max_length=150):  # Increased max_length for better responses
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,  # Slightly increased temperature for more creativity
        top_k=50,  # Top-k sampling for better diversity
        top_p=0.95,  # Nucleus sampling
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit Interface with Context Management
st.title("AI Chatbot Interface")
st.markdown("<style>body {background-color: #f0f2f5;}</style>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

def display_chat():
    for chat in st.session_state.history:
        if isinstance(chat, dict):
            st.markdown(f"<div style='background-color: #e1ffc7; border-radius: 10px; padding: 10px; margin: 5px 0;'>"
                         f"<strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 5px 0;'>"
                         f"<strong>AI:</strong> {chat['ai']}</div>", unsafe_allow_html=True)

# User input for message
user_input = st.text_input("Type your message:", key="user_input", placeholder="Enter your message here...")

# User input for max response length
max_length = st.slider("Select max response length:", min_value=50, max_value=300, value=150)

if user_input:
    try:
        # Combine recent context into a prompt, limiting to the last 4 exchanges
        context = "\n".join(
            [f"You: {c['user']}\nAI: {c['ai']}" for c in st.session_state.history[-4:] if isinstance(c, dict)]
        )
        prompt = f"{context}\nYou: {user_input}\nAI:"
        
        with st.spinner("Generating AI response..."):
            ai_response = generate_text(prompt, max_length=max_length)

        st.session_state.history.append({"user": user_input, "ai": ai_response})

        st.write("### Full Conversation")
        display_chat()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("### Full Conversation")
    display_chat()
