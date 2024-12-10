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


# Function to generate responses based on context
def generate_text(prompt, max_length=150):
    # Tokenize input and move to device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the correct device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate response using GPT-Neo
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and return response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Streamlit Interface with Context Management
st.title("Simple AI Chatbot Interface")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Function to display full conversation history for transparency
def display_chat():
    for chat in st.session_state.history:
        # Display user input
        st.markdown(f"**You:** {chat['user']}")
        # Display AI's response
        st.markdown(f"**AI:** {chat['ai']}")

# Input box for user message
user_input = st.text_input("Type your message:", key="user_input")

if user_input:
    # Generate AI response
    with st.spinner("Generating AI response..."):
        # Combine recent context into a prompt
        context = "\n".join([f"You: {c['user']}\nAI: {c['ai']}" for c in st.session_state.history[-6:]])
        prompt = f"{context}\nYou: {user_input}\nAI:"
        ai_response = generate_text(prompt, max_length=150)

        # Append user input and AI response to history for context tracking
        st.session_state.history.append({"user": user_input, "ai": ai_response})

    # Redisplay chat history
    st.write("### Full Conversation")
    display_chat()
else:
    # If no user input yet, just show context display
    st.write("### Full Conversation")
    display_chat()
