import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import gc

# Load GPT-Neo with Streamlit caching
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Smaller model for CPU usage
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    device = "cpu"  # Use CPU explicitly
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device

# Initialize model and tokenizer
gpt_neo_model, gpt_neo_tokenizer, device = load_gpt_neo()

# Generate text function
def generate_text(prompt, max_length=150):
    inputs = gpt_neo_tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128  # Limit input size for efficiency
    ).to(device)
    
    outputs = gpt_neo_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length + inputs["input_ids"].shape[1],
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=30,
        top_p=0.85,
        pad_token_id=gpt_neo_tokenizer.pad_token_id
    )
    
    gc.collect()  # Clean up unused memory
    return gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

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
    st.cache_resource.clear()
    gc.collect()
    st.success("Cache cleared!")
