import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import re
import random
import time
import json
from datetime import datetime

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Model can be changed to larger if needed
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle padding token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Define helper functions for preprocessing and text manipulation
def clean_input(user_input):
    """Clean user input by removing special characters and excessive spaces."""
    user_input = re.sub(r'[^\w\s]', '', user_input)
    return user_input.strip()

def is_greeting(user_input):
    """Detect if the user is greeting the chatbot."""
    greetings = ["good morning", "good afternoon", "good evening", "hey", "hello", "hi", "howdy"]
    return any(greeting in user_input.lower() for greeting in greetings)

def save_interaction_to_log(user_input, ai_response):
    """Log the user interaction with a timestamp."""
    with open("interaction_log.txt", "a") as f:
        f.write(f"{datetime.now()} - User: {user_input} | AI: {ai_response}\n")

def save_preferences(preferences):
    """Save user preferences to a JSON file."""
    with open("user_preferences.json", "w") as f:
        json.dump(preferences, f)

def load_preferences():
    """Load user preferences from a JSON file."""
    try:
        with open("user_preferences.json", "r") as f:
            preferences = json.load(f)
    except FileNotFoundError:
        preferences = {}
    return preferences

# Function to generate AI response
def generate_text(user_input, max_length=200):
    """Generate AI responses with a context-aware prompt."""
    clean_user_input = clean_input(user_input)
    
    # Check if input is a greeting
    if is_greeting(clean_user_input):
        return random.choice([
            "Hello! How can I assist you today?",
            "Good day! How can I help?",
            "Hi there! What can I do for you?",
            "Greetings! What would you like to discuss?"
        ])
    
    # Build conversation context
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history[-5:]]
    )
    
    prompt = (
        "You are a friendly, intelligent, and informative AI assistant. Respond in a manner suitable "
        "to the conversation context. If the user asks a question, provide a detailed, informative response."
        f"\n{conversation_context}\nUser: {clean_user_input}\nAI:"
    )
    
    # Tokenize input and generate AI response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)

    # Generate AI output
    outputs = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = generated_text.find("AI:") + len("AI:")
    response = generated_text[response_start:].strip()

    # Log the interaction
    save_interaction_to_log(user_input, response)
    
    return response

# Initialize Streamlit state
if "history" not in st.session_state:
    st.session_state.history = []

# Load user preferences from file (if available)
user_preferences = load_preferences()

# Streamlit UI setup
st.title("Advanced AI Chatbot")
st.markdown("""
This is an advanced AI chatbot that provides detailed responses, personalizes conversation, 
and allows dynamic adjustments to the tone, topic, and style of interaction.
""")

# User preferences for customization
if st.checkbox("Enable Custom Tone"):
    tone_option = st.selectbox("Choose the tone for responses:", ["Friendly", "Professional", "Casual", "Empathetic"])
    user_preferences["tone"] = tone_option
    save_preferences(user_preferences)

if st.checkbox("Enable Topic Focus"):
    topic_option = st.selectbox("Choose a topic:", ["Science", "Technology", "History", "Music", "General Knowledge"])
    user_preferences["topic"] = topic_option
    save_preferences(user_preferences)

# Input box for user to type messages
with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", placeholder="Ask me anything...")
    submit_button = st.form_submit_button(label="Send")

# Process and respond to user input
if submit_button and user_input:
    with st.spinner("Generating response..."):
        ai_response = generate_text(user_input)  # Get AI response
        st.session_state.history.append({"user": user_input, "ai": ai_response})  # Save to history

# Display conversation history
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.history = []

# Feedback and suggestions section
st.markdown("""
### Suggestions for improving the chatbot:
- You can switch topics at any time.
- Use the "Enable Custom Tone" checkbox to change how the AI responds.
- Feel free to ask for more detailed answers or summaries.
""")

# Saving preferences, interaction log, and other settings
if st.button("Save Preferences"):
    save_preferences(user_preferences)
    st.success("Your preferences have been saved successfully.")

# Adding multi-turn conversation handling with advanced user input analysis
def analyze_input(user_input):
    """Analyze user input to detect intent, entities, and possible topics."""
    # Example of detecting if the user asks for a recommendation
    if "recommend" in user_input.lower():
        return "recommendation", None
    
    # Check if user asks for help with a specific topic
    if "help" in user_input.lower():
        return "help", "Please specify a topic"
    
    return "general", "I'm here to chat about anything!"

# Detect user intent and respond accordingly
def handle_advanced_input(user_input):
    intent, message = analyze_input(user_input)
    if intent == "recommendation":
        return "I recommend reading some books or exploring online resources related to your topic."
    elif intent == "help":
        return message
    return "How can I assist you further?"

# Enhance response customization
if user_input:
    advanced_response = handle_advanced_input(user_input)
    st.markdown(f"**AI (Advanced Handling):** {advanced_response}")
