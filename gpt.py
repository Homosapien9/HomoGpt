import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Generate AI response with enhanced logic
def generate_text(user_input, max_length=150):
    """
    Generates an AI response based on user input with strict relevance and clarity.
    """
    # Include only the last 3 interactions for context
    context_window = 3
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history[-context_window:]]
    )

    # Define a clear, concise prompt
    prompt = (
        "You are a helpful and knowledgeable assistant. Answer user queries with concise and relevant information.\n\n"
        f"{conversation_context}\nUser: {user_input}\nAI:"
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)

    # Generate response with controlled parameters
    outputs = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.5,  # Lower temperature for focused responses
        top_k=40,  # Limits token selection to top 40 options
        top_p=0.8,  # Ensures nucleus sampling for balanced output
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode output and extract AI response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = generated_text.find("AI:") + len("AI:")
    response = generated_text[response_start:].strip()

    # Avoid hallucination by truncating at the first newline or stopping criteria
    response = response.split("\n")[0].strip()
    return response

# Streamlit UI
st.title("Advanced AI Chatbot")
st.markdown("Chat with a highly responsive and intelligent assistant.")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", placeholder="Ask anything...")
    submit_button = st.form_submit_button(label="Send")

# Process user input and AI response
if submit_button and user_input:
    with st.spinner("Thinking..."):
        ai_response = generate_text(user_input)  # Generate AI response
        st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state.history = []
