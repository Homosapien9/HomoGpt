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
        
        # Set pad_token to eos_token to fix padding-related issues
        tokenizer.pad_token = tokenizer.eos_token
        
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
        # Tokenize input
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
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode and return text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        st.error(f"Error during text generation: {str(e)}")
        print(e)
        return "An error occurred while generating text."


# Streamlit Web Interface
st.title("AI-Powered Chatbot with GPT-Neo")
st.write("Chat with GPT-Neo! Ask anything you'd like.")

# Load GPT model
model, tokenizer, device = load_gpt_neo()

# Check if GPT model was loaded successfully
if model and tokenizer:
    # User input with memory for context
    if "history" not in st.session_state:
        st.session_state.history = []
    
    user_input = st.text_input("You:", key="user_input")
    
    if user_input:
        with st.spinner("Thinking..."):
            # Append user input to conversation history
            st.session_state.history.append(f"You: {user_input}")
            
            # Create conversation history as context
            context = "\n".join(st.session_state.history[-6:])  # Keep the last few exchanges
            
            # Generate response
            response = generate_text(context, model, tokenizer, device, max_length=200)
            
            # Update history with the assistant's response
            st.session_state.history.append(f"AI: {response}")
            
            # Display the entire conversation
            st.write("\n".join(st.session_state.history))
else:
    st.error("Could not load GPT-Neo. Ensure internet connection or check server setup.")
