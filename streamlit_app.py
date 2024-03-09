# Import necessary libraries
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load and preprocess data
# Assuming df is a DataFrame containing the cleaned text
# Make sure df is defined or loaded correctly before using it in Streamlit components

# Streamlit app
st.title("Memory Support Chatbox for Pregnant Women")
user_input = st.text_input("You:", "Enter your message here...")
if user_input:
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    reply_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    st.text_area("Chatbot:", value=reply_text, height=200)

# Text Analysis
st.subheader("Text Analysis")

# Word Cloud
st.subheader("Word Cloud")
all_text = " ".join(df["cleaned_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Sample Prompts
st.subheader("Sample Prompts")
sample_prompts = [
    "What causes pregnancy brain fog?",
    "How does pregnancy affect the brain?",
    "How can I improve my memory during pregnancy?",
    "Can pregnancy brain fog affect my ability to work or perform daily tasks?",
]

for prompt in sample_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(f"**Prompt:** {prompt}\n**Response:** {response}\n")

# Displaying DataFrame
st.subheader("DataFrame")
st.write(df)

