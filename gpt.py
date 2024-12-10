import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # You can use a larger model if needed
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Function to generate responses based on context
def generate_text(prompt, max_length=150):  # Set max_length to 150 for concise responses
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,  # Lower temperature for more focused responses
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

# Streamlit Interface
st.title("General GPT Chatbot")
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f0f5;  /* Light background for better readability */
    }
    .chat-container {
        background-color: #ffffff;  /* White background for chat messages */
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

# Display the chat history
def display_chat():
    if st.session_state.history:
        last_chat = st.session_state.history[-1]  # Get the last chat
        st.markdown(f"<div class='chat-container'><strong>You:</strong> {last_chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-container'><strong>AI:</strong> {last_chat['ai']}</div>", unsafe_allow_html=True)

# User input for message
user_input = st.text_input("Type your message:", key="user_input", placeholder="Enter your message here...")

if user_input:
    # Construct the prompt
    prompt = f"You are a helpful assistant. Answer the following question:\n:User  {user_input}\nAI:"

    with st.spinner("Generating AI response..."):
        ai_response = generate_text(prompt)  # Generate the AI response

    # Append the user input and AI response to the history
    st.session_state.history.append({"user": user_input, "ai": ai_response})

    # Display the latest query and response
    display_chat()

# Display the chat history at the top
display_chat()
