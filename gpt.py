import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

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

def generate_text(user_input, max_length=200):
    """
    Generate AI response using GPT-Neo.
    """
    # Limit context to avoid confusion
    context_window = 5  # Keep only the last 5 exchanges
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history[-context_window:]]
    )
    
    # Construct prompt with clear instructions
    prompt = (
        "The following is a friendly and helpful conversation between a user and an AI assistant. "
        "The AI provides clear, natural language responses and avoids generating code unless explicitly requested.\n"
        f"{conversation_context}\nUser: {user_input}\nAI:"
    )

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)

    # Generate response
    outputs = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AI response
    response_start = generated_text.find("AI:") + len("AI:")
    return generated_text[response_start:].strip()

# Streamlit Interface
st.title("AI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", placeholder="Ask me anything...")
    submit_button = st.form_submit_button(label="Enter")

if submit_button and user_input:
    with st.spinner("Generating response..."):
        ai_response = generate_text(user_input)
        st.session_state.history.append({"user": user_input, "ai": ai_response})

if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

if st.button("Clear Chat"):
    st.session_state.history = []
