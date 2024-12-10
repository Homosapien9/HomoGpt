import streamlit as st
import PyPDF2
import docx
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


# Load GPT-Neo model and tokenizer with caching
@st.cache_resource
def load_gpt_neo():
    try:
        model_name = "EleutherAI/gpt-neo-1.3B"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        
        # Decide device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading GPT-Neo model: {str(e)}")
        print(e)
        return None, None, None


# Function to generate text based on GPT-Neo
def generate_text(prompt, model, tokenizer, device, max_length=150):
    try:
        # Tokenize input and prepare for generation
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Ensure input is moved to the correct device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Perform text generation
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode and return text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        st.error(f"Error during text generation: {str(e)}")
        print(e)
        return "An error occurred while generating text."


# Streamlit Web Interface
st.title("AI Summarizer and Chatbot")
st.write("Type a message, and I will generate a response for you!")

# Load GPT model
model, tokenizer, device = load_gpt_neo()

# Check if GPT model was loaded successfully
if model and tokenizer:
    # Text input field for the user
    user_input = st.text_input("Enter your message:")
    
    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Generate response
                result = generate_text(user_input, model, tokenizer, device, max_length=500)
                
                # Show generated response
                st.write("Generated Response:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                print(e)
else:
    st.error("Could not load GPT-Neo. Check internet connection or server configurations.")
