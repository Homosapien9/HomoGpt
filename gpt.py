import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer with caching
@st.cache_resource
def load_gpt_neo():
    try:
        model_name = "EleutherAI/gpt-neo-125M"  # Use a smaller model for faster response
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading GPT-Neo model: {str(e)}")
        print(e)
        return None, None, None

# Function to generate text based on GPT-Neo
def generate_text(prompt, model, tokenizer, device, max_length=100):  # Reduced max_length
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.9,  # Adjusted temperature for more diverse responses
            pad_token_id=tokenizer.pad_token_id,
        )
        
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

if model and tokenizer:
    if "history" not in st.session_state:
        st.session_state.history = []
    
    user_input = st.text_input("You:", key="user_input")
    
    if user_input:
        with st.spinner("Thinking..."):
            st.session_state.history.append(f"You: {user_input}")
            
            # Create conversation history as context, limiting to the last 4 exchanges
            context = "\n".join(st.session_state.history[-4:])  # Reduced to last 4 exchanges
            
            response = generate_text(context, model, tokenizer, device, max_length=100)  # Reduced max_length
            
            st.session_state.history.append(f"AI: {response}")
            st.write("\n".join(st.session_state.history))
else:
    st.error("Could not load GPT-Neo. Ensure internet connection or check server setup.")
