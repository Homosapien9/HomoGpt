import streamlit as st
import PyPDF2
import docx
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

def generate_text(prompt, max_length=150):
    # Tokenize input and add padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the correct device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate text
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit Web Interface
st.title("AI Summarizer and Chatbot")
user_input = st.text_input("Enter your message:")

if user_input:
    with st.spinner("Generating response..."):
        response = generate_text(user_input, max_length=150)
        st.write("AI Response:")
        st.write(response)
