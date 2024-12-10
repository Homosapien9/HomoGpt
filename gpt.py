import streamlit as st
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
st.title("AI Chatbot")

# Initialize chat history in session state if it's not already initialized
if "history" not in st.session_state:
    st.session_state.history = []

# Function to display the full chat history in a user-readable format
def display_chat():
    for chat in st.session_state.history:
        # Ensure each chat is a dictionary with user and ai keys before displaying
        if isinstance(chat, dict):
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**AI:** {chat['ai']}")


# Input box for user message
user_input = st.text_input("Type your message:", key="user_input")

if user_input:
    try:
        # Combine recent context into a prompt
        context = "\n".join(
            [f"You: {c['user']}\nAI: {c['ai']}" for c in st.session_state.history[-6:] if isinstance(c, dict)]
        )
        prompt = f"{context}\nYou: {user_input}\nAI:"
        
        # Generate AI response
        with st.spinner("Generating AI response..."):
            ai_response = generate_text(prompt, max_length=150)

        # Append user input and AI response to history safely
        st.session_state.history.append({"user": user_input, "ai": ai_response})

        # Redisplay the conversation history
        st.write("### Full Conversation")
        display_chat()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    # If no user input yet, show context only
    st.write("### Full Conversation")
    display_chat()
