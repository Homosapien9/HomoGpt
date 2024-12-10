import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Specify GPT-Neo model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Generate AI response with advanced prompt engineering
def generate_text(user_input, max_length=150):
    """
    Generates an AI response based on the user's input with improved context handling and response generation.
    """
    # Maintain context of the last N messages for relevance
    context_window = 5  # Limit to last 5 interactions
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history[-context_window:]]
    )

    # Advanced prompt for clear, concise, and focused responses
    prompt = (
        "The following is a conversation with a helpful and knowledgeable AI assistant. "
        "The AI provides accurate, concise, and context-aware answers to user queries. "
        "It avoids unnecessary elaboration and only provides code or technical details if explicitly requested.\n\n"
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
        temperature=0.6,  # Lower temperature for deterministic responses
        top_k=40,  # Focus on top 40 probable tokens
        top_p=0.9,  # Use nucleus sampling for balanced responses
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode output and extract AI response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = generated_text.find("AI:") + len("AI:")
    return generated_text[response_start:].strip()

# Streamlit UI for Chatbot
st.title("Advanced AI Chatbot")
st.markdown(
    """
    Welcome to your advanced AI chatbot! This assistant provides accurate, context-aware, and concise responses to your queries.
    """
)

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Input form for user interaction
with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", placeholder="Ask me anything...")
    submit_button = st.form_submit_button(label="Send")

# Handle user input and AI response
if submit_button and user_input:
    with st.spinner("Thinking..."):
        ai_response = generate_text(user_input)  # Generate AI response
        # Append to conversation history
        st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display conversation history
if st.session_state.history:
    st.markdown("### Chat History")
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state.history = []

