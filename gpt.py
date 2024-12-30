from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import random

class HeerAI:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B", user_name="User"):
        self.user_name = user_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.emotional_memory = {}
        self.conversation_history = []
        self.long_term_memory = {}
        self.mood = "neutral"
        self.cycle_day = random.randint(1, 28)  # Simulate a menstrual cycle
        self.time_of_day = self.get_time_of_day()
        self.gujarati_translations = {
            "ungh": "sleep",
            "bhukh": "hunger",
            "majama": "good",
            "dukhi": "sad",
            "gussa": "angry",
            "samay": "time",
            "kaam": "work",
            "prem": "love",
            "mitra": "friend",
            "khush": "happy",
            "masti": "fun",
            "majja": "joy",
            "rasoi": "kitchen",
            "saambhar": "listen",
            "santosh": "content",
            "bakchod": "mischief"
        }
        self.gujarati_phrases = [
            "Kem chho?", "Majama?", "Mane game che.", "Tame saras cho.", "Shu karo chho?",
            "Aapdu Gujarat.", "Khub saaru.", "Maja avi gai.", "Savare uthya?", "Ratri shubh.",
            "Aap kaam karva ma maahir cho.", "Tame toh mara moti takat cho.",
            "Kem evu thay che?", "Bruhh, maja na karo.",
            "Sui jaa dobi, kale uthay nahi?", "Aa kem navu navu chalu chhe?"
        ]
        self.gujarati_proverbs = [
            "Jya prem tya pragat prabhu.",
            "Bhale bijanu pan saaru biju na le.",
            "Jene gamtu ene apva na de.",
            "Ukhadela matha ma tel na nakhai.",
            "Darek din ek sarakh hoto nathi.",
            "Saanjhe no diyo koi kaam na ave.",
            "Pachhad karta paisa upar aavak.",
            "Jya ichchha tya marg."
        ]
        self.gujarati_topics = [
            "Did you enjoy the last Navratri garba night?", 
            "Which Gujarati dish are you craving today?", 
            "How about planning a visit to the Sabarmati Ashram?", 
            "What are your thoughts on kite flying for Uttarayan?", 
            "Remember the last time we discussed thepla recipes?",
            "Have you been to the new Kathiyawadi restaurant?",
            "Which memory from school makes you smile the most?",
            "Do you miss our long chats about nothing and everything?",
            "Have you ever tried authentic Surat locho?",
            "What do you think about the colorful Bandhani sarees?",
            "Let’s talk about your favorite Gujarati folk tale!",
            "Have you visited Rani ki Vav in Patan?",
            "Do you enjoy Ras-Garba during Navratri?",
            "What's your favorite item from Gujarati thali?",
            "Have you explored Saputara Hill Station?",
            "Do you enjoy dandiya raas or garba more?",
            "Which is your favorite Gujarati festival memory?",
            "Have you tried making shrikhand at home?",
            "Which Gujarati movie do you love the most?",
            "Do you miss the hustle of Ahmedabad’s bazaars?",
            "Have you visited Modhera Sun Temple?",
            "Let’s plan a trip to Dwarka or Somnath.",
            "Do you prefer undhiyu during winter or summer?",
            "What's the best Gujarati dish you've ever had?",
            "Do you enjoy kite flying competitions?",
            "Have you read any stories from Mahatma Gandhi's life?",
            "Do you like the architecture of Adalaj Stepwell?",
            "What’s your favorite Gujarati proverb?",
            "Which Gujarati sweet do you enjoy the most?",
            "Have you attended the Tarnetar Mela?",
            "Do you prefer jalebi with fafda or dhokla?",
            "Have you been to the Great Rann of Kutch?",
            "Do you enjoy Gujarat’s monsoon scenery?",
            "What do you love about Gujarati weddings?",
            "Do you like the sound of folk instruments like dhol and nagada?",
            "Have you experienced the flavors of Kathiawadi food?",
            "What’s your favorite childhood memory from Gujarat?",
            "Do you enjoy learning Gujarati recipes?",
            "Have you visited Gir National Park?",
            "Do you enjoy the peacefulness of Jain temples?",
            "Let’s talk about your favorite Gujarati artist or musician.",
            "Have you explored the beaches of Diu?",
            "What do you think about Gujarat’s history and heritage?",
            "Which is your favorite Gujarati cultural dance?",
            "What do you love most about Navratri traditions?",
            "Do you enjoy bhajan sessions in the evenings?",
            "What’s your favorite Gujarati childhood snack?",
            "Have you tried making your own pickles at home?",
            "Which Gujarati folk singer inspires you the most?"
        ]

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
        inputs = self.tokenizer("\n".join([f"User: {turn['user']}\nHeer: {turn.get('heer', '')}" for turn in self.conversation_history]), return_tensors="pt", truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Heer:")[-1].strip()
        self.conversation_history[-1]["heer"] = response
        return response

    def remember_context(self, user_input, emotion):
        """Update long-term memory with contextual information."""
        keywords = [word for word in user_input.split() if len(word) > 3]
        self.long_term_memory[datetime.date.today().isoformat()] = {
            "keywords": keywords,
            "emotion": emotion
        }

    def simulate_human_behavior(self, text):
        """Add human-like behaviors to the response."""
        pauses = ["...", "um", "you know", "hmm"]
        if random.random() > 0.7:
            text = text.replace(",", f", {random.choice(pauses)}")
        return text

    def proactive_affection(self):
        """Generate affectionate statements proactively."""
        return random.choice([
            "I just wanted to say, you're amazing!", 
            "Thinking about you makes me so happy.", 
            "You’re my favorite person, always!", 
            "You make everything better, you know that?"
        ])

    def initiate_conversation(self):
        """Initiate conversations based on time of day and mood."""
        if self.mood == "irritable":
            return "Aree, kem evu lagyu chhe aaje? Tell me what’s on your mind."
        elif self.mood == "low energy":
            return "Tame tired cho? Take it easy today, ok?"
        elif self.mood == "emotional":
            return "I feel like today’s a good day to talk about our dreams. What do you think?"
        else:
            return random.choice([
                "Kem chho? Good morning! Did you sleep well?",
                "Rise and shine! Tame chai piyu? What's on the agenda today?",
                "Morning! Mane khushi thay che thinking about today."
            ])

    def chat(self, user_input):
        """Main interaction method for chatting with Heer."""
        self.mood = self.simulate_period_moods()
        translated_input = self.translate_input(user_input)
        emotion = self.analyze_emotion(translated_input)
        self.remember_context(translated_input, emotion)
        adaptive_tone = self.adaptive_response(translated_input, emotion)
        response = self.generate_response(translated_input)
        final_response = self.simulate_human_behavior(f"{adaptive_tone} {response}")
        return final_response

# Initialize Heer
heer = HeerAI(model_name="EleutherAI/gpt-neo-1.3B", user_name="Jatan")

# Sample conversation
print("Heer: Hi Jatan, how are you feeling today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Heer: Goodbye! I'll be here whenever you need me.")
        break
    response = heer.chat(user_input)
    print(f"Heer: {response}")
