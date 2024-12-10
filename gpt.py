import streamlit as st
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load GPT-Neo model and tokenizer
@st.cache_resource
def load_gpt_neo():
    model_name = "EleutherAI/gpt-neo-125M"  # Specify the GPT-Neo model to use
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle padding token
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_gpt_neo()

# Function to generate AI response based on user input
def generate_text(user_input, max_length=200):
    """
    Generate AI response based on user input.
    """
    # Limit context to the last few exchanges to ensure relevant responses
    max_context_tokens = 500  # Limit for the total tokens in context
    conversation_context = "\n".join(
        [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.history]
    )

    # If the context exceeds the token limit, summarize it
    if len(tokenizer.encode(conversation_context)) > max_context_tokens:
        conversation_context = summarize_context(conversation_context)

    # Construct the prompt
    prompt = f"{conversation_context}\nUser: {user_input}\nAI:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate response
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Prevent repetitive phrases
        temperature=0.8,  # Creativity control
        top_k=40,  # Encourage diverse word choice
        top_p=0.95,  # Nucleus sampling for controlled randomness
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract AI's response after "AI:"
    response_start = generated_text.find("AI:") + len("AI:")
    return generated_text[response_start:].strip()

# Summarization function for long contexts
def summarize_context(context):
    """
    Summarize long conversation contexts to maintain relevance.
    """
    summary_prompt = f"Summarize this conversation:\n{context}"
    inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)

    summary_output = model.generate(
        input_ids,
        max_length=100,  # Keep the summary short and relevant
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    summary = tokenizer.decode(summary_output[0], skip_special_tokens=True)
    return summary.strip()

# Streamlit Interface
st.title("AI Chatbot")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Input field with an "Enter" button
with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", placeholder="Ask me anything...")
    submit_button = st.form_submit_button(label="Enter")

# Process user input
if submit_button and user_input:
    with st.spinner("Generating response..."):
        ai_response = generate_text(user_input)  # Generate AI response
        st.session_state.history.append({"user": user_input, "ai": ai_response})

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")

# Clear chat history button
if st.button("Clear Chat"):
    st.session_state.history = []
