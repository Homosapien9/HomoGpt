import streamlit as st
import PyPDF2
import docx
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from pptx import Presentation
from pptx.util import Inches
import requests
from io import BytesIO

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to generate text using GPT-Neo (Summarization)
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to get an image from an API like Unsplash (optional)
def get_image_url(query):
    UNSPLASH_ACCESS_KEY = 'cO4ExsQJ3elT0pNYXcGwGs7qgSRuq69U5qVWIJXqTkg'
    UNSPLASH_URL = 'https://api.unsplash.com/photos/random'
    params = {'query': query, 'client_id': UNSPLASH_ACCESS_KEY, 'count': 1}
    response = requests.get(UNSPLASH_URL, params=params)
    if response.status_code == 200:
        image_url = response.json()[0]['urls']['regular']
        return image_url
    return None

# Function to create a PowerPoint presentation
def create_ppt(summary):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content layout
    title = slide.shapes.title
    title.text = "Summary of Document"
    
    content = slide.shapes.placeholders[1]
    content.text = summary

    # Fetch and add an image based on a keyword (optional)
    query = summary.split()[0]  # You can use any logic for querying
    image_url = get_image_url(query)

    if image_url:
        image_response = requests.get(image_url)
        image_stream = BytesIO(image_response.content)
        slide.shapes.add_picture(image_stream, Inches(1), Inches(1.5), width=Inches(5), height=Inches(3))

    ppt_filename = "generated_presentation.pptx"
    prs.save(ppt_filename)
    return ppt_filename

# Streamlit Web Interface
st.title("AI-Powered Document Summarizer and PowerPoint Generator")
st.write("Upload a file, and let AI summarize its contents and generate a PowerPoint presentation.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text from the uploaded file
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    
    st.write("Extracted Text:")
    st.write(text[:1000])  # Display the first 1000 characters of the extracted text for preview

    # Summarize text using GPT-Neo
    if st.button("Summarize and Generate PPT"):
        if text:
            with st.spinner("Summarizing and generating PowerPoint..."):
                summary = generate_text(text, max_length=500)  # Summarize the document content
                st.write("Summary Generated:")
                st.write(summary)

                # Create PowerPoint presentation based on the summary
                ppt_filename = create_ppt(summary)
                st.success("PowerPoint generated successfully!")
                st.download_button("Download PowerPoint", ppt_filename, file_name="generated_presentation.pptx")
        else:
            st.error("No content to summarize.")
