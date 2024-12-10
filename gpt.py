import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import wikipedia

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-1.3B"  # You can choose different sizes like 1.3B, 2.7B, or 125M
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle padding token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Function to generate AI response based on user input
def generate_text(user_input, max_length=150):
    # Build conversation context (only user and AI prompts)
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history]
    )
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

# Fetch Wikipedia information for a given query
def get_wikipedia_info(query):
    try:
        # Fetch the summary of the query from Wikipedia
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # In case of ambiguity, return a list of possible options
        return f"There are multiple topics for '{query}'. Please be more specific."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "Failed to fetch information from Wikipedia. Try again later."

# Streamlit Interface
st.title("AI Chatbot")
st.markdown("This is an advanced AI chatbot powered by GPT-Neo and Wikipedia.")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# User input for message
user_input = st.text_input("Your message:", placeholder="Ask me anything...")

# Process user input and generate AI response
if user_input:
    if user_input.lower().startswith("who is") or user_input.lower().startswith("what is"):
        # If the query is related to a fact, use Wikipedia
        with st.spinner("Fetching information from Wikipedia..."):
            wikipedia_info = get_wikipedia_info(user_input)
            st.session_state.history.append({"user": user_input, "ai": wikipedia_info})
    else:
        # Otherwise, generate response using GPT-Neo
        with st.spinner("Generating response..."):
            ai_response = generate_text(user_input)  # Generate AI response
            st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

# Clear chat history
if st.button("Clear Chat"):
    st.session_state.history = []
