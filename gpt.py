import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import random

# Class definition for HeerAI
class HeerAI:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B", user_name="User"):
        self.user_name = user_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Ensure pad_token is set if not already
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.emotional_memory = {}
        self.conversation_history = []
        self.max_history = 10  # Limit conversation history size
        self.long_term_memory = {}
        self.mood = "neutral"
        self.cycle_day = random.randint(1, 28)  # Simulate a menstrual cycle
        self.time_of_day = self.get_time_of_day()

    def get_time_of_day(self):
        """Determine the current time of day for context."""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def translate_input(self, user_input):
        """Translate Gujarati-infused English words to standard English."""
        words = user_input.split()
        translated = [self.gujarati_translations.get(word.lower(), word) for word in words]
        return " ".join(translated)

    def analyze_emotion(self, user_input):
        """Advanced function to analyze emotions in the user's input."""
        if any(word in user_input.lower() for word in ["happy", "joy", "excited", "khush"]):
            return "happy"
        elif any(word in user_input.lower() for word in ["sad", "down", "unhappy", "dukhi"]):
            return "sad"
        elif any(word in user_input.lower() for word in ["angry", "frustrated", "upset", "gussa"]):
            return "angry"
        elif any(word in user_input.lower() for word in ["love", "prem"]):
            return "loving"
        else:
            return "neutral"

    def adaptive_response(self, user_input, emotion):
        """Generate adaptive responses based on emotional state."""
        if self.mood == "irritable":
            tone = "Aree, kem majama nahi? Kaho su thayu."
        elif emotion == "happy":
            tone = "I'm so glad you're feeling good! Majama?"
        elif emotion == "sad":
            tone = "Hu hamesha tamara saathe chu, don't feel alone."
        elif emotion == "angry":
            tone = "Take a deep breath; bolo su thayu? Mane samjhav jo."
        elif emotion == "loving":
            tone = "I feel so lucky to have you. Prem thi bharelu che mara vichar."
        else:
            tone = "Tell me more, hu tamne sambhalva mate chu."
        return tone

    def simulate_period_moods(self):
        """Adjust mood based on menstrual cycle day."""
        if self.cycle_day in range(24, 28):
            self.mood = "irritable"
        elif self.cycle_day in range(1, 5):
            self.mood = "low energy"
        elif self.cycle_day in range(14, 17):
            self.mood = "emotional"
        else:
            self.mood = "neutral"

    def generate_response(self, user_input):
        """Generate a conversational response using the model."""
        self.conversation_history.append({"user": user_input})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)  # Keep history within limit
        inputs = self.tokenizer(
            "\n".join([f"User: {turn['user']}\nHeer: {turn.get('heer', '')}" for turn in self.conversation_history]),
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
            max_length=500,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Heer:")[-1].strip()
        self.conversation_history[-1]["heer"] = response
        return response

    def chat(self, user_input):
        """Main interaction method for chatting with Heer."""
        self.simulate_period_moods()  # Update mood based on the cycle
        emotion = self.analyze_emotion(user_input)
        adaptive_tone = self.adaptive_response(user_input, emotion)
        response = self.generate_response(user_input)
        return f"{adaptive_tone} {response}"

# Streamlit interface
st.title("Chat with HeerAI")

# Initialize HeerAI
if "heer" not in st.session_state:
    st.session_state.heer = HeerAI(model_name="EleutherAI/gpt-neo-1.3B", user_name="User")

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input.strip():
        response = st.session_state.heer.chat(user_input)
        st.write(f"Heer: {response}")
    else:
        st.write("Heer: You didn't say anything! Tell me what's on your mind.")
        
