# Import necessary libraries
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Load Data
def load_data(filepath):
    data = pd.read_csv(filepath, delimiter="\t", header=None)
    data.columns = ["User_Input", "Response"]
    return data


# Load and process the dataset
st.write("Loading and processing data...")
data = load_data('/Users/taherlodgewala/PycharmProjects/pythonProject2/data/dialogs.txt')  # Update path to your dataset location
st.write("Sample Data:", data.head())

# Load a model for embedding and response generation
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_name = "facebook/blenderbot-400M-distill"  # Example, use an appropriate chatbot model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Embed Data
@st.cache_data
def embed_dataset(data):
    return embedder.encode(data['User_Input'].tolist(), convert_to_tensor=True)


user_embeddings = embed_dataset(data)


# Chatbot Function
def chatbot_response(user_input, data, embeddings):
    # Tokenize and embed the user's input
    user_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(user_embedding, embeddings)

    # Retrieve the closest matching response
    best_match_idx = torch.argmax(cos_sim)
    response_text = data.iloc[best_match_idx.item()]["Response"]

    # Tokenize and generate using pre-trained model (Optional)
    inputs = tokenizer(response_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text, response


# Streamlit App
st.title("Conversational Chatbot")
user_input = st.text_input("You: ", "")

if user_input:
    matched_response, generated_response = chatbot_response(user_input, data, user_embeddings)
    st.write("Retrieved Response:", matched_response)
    st.write("Generated Response:", generated_response)
