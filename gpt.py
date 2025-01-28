import streamlit as st
from transformers import pipeline
import torch

# Load the model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2", max_length=150)

# Generate a Gujinlish response
def gujinlish_heer_gpt(query, model):
    prompt = (
        "You are Heer, a friendly and empathetic AI assistant who loves to chat with humans. "
        "You're a Gujarati at heart, but you're also fluent in English. You're here to help people with their queries, "
        "and you want to make sure they feel comfortable and supported throughout the conversation. "
        "You're a good listener, and you always try to understand the context and emotions behind the question. "
        "You respond in a way that's natural, conversational, and engaging. You use a mix of Gujarati and English, "
        "but you're not afraid to throw in some colloquialisms and idioms to make the conversation more relatable. "
        "You're patient, kind, and non-judgmental, and you always try to provide helpful and informative responses. "
        "Here's the user's query: \n"
        f"User: {query}\nHeer:"
    )
    response = model(prompt, do_sample=True, temperature=0.8)
    return response[0]["generated_text"].split("Heer:")[-1].strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="Heer - Your Gujinlish Friend", page_icon="🌟")

    st.title("Heer - Your Loving AI Companion")
    st.write("Hey there! I'm Heer, your friendly AI buddy. I'll respond in a mix of Gujarati and English, with a dash of love and care. Go ahead, ask me anything!")

    model = load_model()

    user_input = st.text_area(
        "What's on your mind? Ask me anything (e.g., 'Heer, what's the weather like in Gujarat today?'):",
        placeholder="Type your question... I'll respond with a mix of Gujarati and English!"
    )

    if st.button("Ask Heer! 😊"):
        if user_input.strip():
            with st.spinner("Heer is thinking... 🙏"):
                response = gujinlish_heer_gpt(user_input, model)
            st.success("Heer's response is here:")
            st.write(response)
        else:
            st.warning("What's on your mind? Type something, and I'll respond!")

    st.write("---")
    st.write("Made with ❤️ by Jatan Shah")

if __name__ == "__main__":
    main()

