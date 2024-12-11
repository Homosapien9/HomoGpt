import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Check if CUDA is available (GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GPT-Neo model and tokenizer with caching
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Using a smaller model for CPU
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    # Load the model and send it to the correct device (CPU or GPU)
    model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device

# Initialize the model and tokenizer
gpt_neo_model, gpt_neo_tokenizer, device = load_gpt_neo()

# Generate text function
def generate_text(prompt, max_length=150):
    # Tokenize input
    inputs = gpt_neo_tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128  # Input size limit for efficiency
    ).to(device)  # Send input to correct device (GPU or CPU)

    # Generate the output
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
    
    return gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# Streamlit Interface
st.title("GPT-Neo Powered AI Chatbot")
st.markdown("Generate text and handle tasks efficiently using GPT-Neo. This app uses GPU if available!")

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
    st.success("Cache cleared!")
