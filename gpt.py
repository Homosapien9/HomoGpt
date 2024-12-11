import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo with Streamlit caching
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Use smaller model for faster response times
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    device = "cpu"  # Use CPU explicitly
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device

# Initialize model and tokenizer
gpt_neo_model, gpt_neo_tokenizer, device = load_gpt_neo()

# Create a memory cache for conversation history
conversation_history = []

# Generate text function with improved settings
def generate_text(prompt, max_length=150):
    # Add prompt to conversation history
    conversation_history.append(f"User: {prompt}")
    
    # Prepare the input for the model, combining history for context
    full_input = "\n".join(conversation_history)
    
    inputs = gpt_neo_tokenizer(
        full_input,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512  # Increase context window
    ).to(device)
    
    # Generate the response with parameters that simulate a more human-like style
    outputs = gpt_neo_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + inputs["input_ids"].shape[1],
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,  # Make responses more creative
        top_k=50,  # Increase diversity of responses
        top_p=0.9,  # Use nucleus sampling
        pad_token_id=gpt_neo_tokenizer.pad_token_id,
        eos_token_id=gpt_neo_tokenizer.eos_token_id
    )
    
    response = gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add the response to conversation history to keep the context
    conversation_history.append(f"AI: {response.strip()}")
    
    return response.strip()

# Streamlit Interface
st.title("GPT-Neo Powered AI Chatbot")
st.markdown("Generate text and handle tasks efficiently using GPT-Neo on CPU.")

# User Input
user_input = st.text_area("Enter your prompt:")

if user_input:
    with st.spinner("Processing..."):
        try:
            response = generate_text(user_input)
            st.write("### AI Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Clear Cache Button
if st.button("Clear Cache"):
    # Clear conversation history and Streamlit cache
    global conversation_history
    conversation_history = []
    st.cache_resource.clear()
    st.success("Cache cleared!")
