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

# Function to generate responses based on context and advanced prompt engineering
def generate_text(user_input, max_length=150):
    # Maintain context for better responses
    context = "\n".join([f":User  {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history])
    
    # Advanced prompt engineering
    prompt = (
        "You are a helpful assistant that provides detailed and informative responses.\n"
        "Always clarify your answers and provide examples when necessary.\n"
        "If the user asks for code, provide well-structured code snippets with comments.\n"
        "If the user asks for explanations, break down complex concepts into simpler parts.\n"
        f"{context}\n"
        f":User  {user_input}\n"
        "AI:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),  # Adjust max_length based on input length
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,  # Adjusted for more creative responses
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

# Streamlit Interface
st.title("Advanced AI Chatbot")
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f0f0;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .user-message {
        background-color: #d1e7dd;
        border-left: 5px solid #0d6efd;
    }
    .ai-message {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

# User input for message
user_input = st.text_input("Type your message:", key="user_input", placeholder="Enter your message here...")

if user_input:
    with st.spinner("Generating AI response..."):
        ai_response = generate_text(user_input)  # Generate the AI response

    # Append the user input and AI response to the history
    st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display the chat history at the top
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"<div class='chat-container user-message'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-container ai-message'><strong>AI:</strong> {chat['ai']}</div>", unsafe_allow_html=True)

# Clear button to reset the chat history
if st.button("Clear Chat"):
    st.session_state.history = []
