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

txt_files = [
    "Cognition in Pregnancy- Perceptions and Performance, 2005-2006 - Dataset - B2FIND.txt",
    "Frontiers | Cognitive disorder and associated factors among pregnant women attending antenatal servi.txt",
    "Frustrated By Brain Fog? How Pregnancy Actually Alters Yo....txt",
    "Is Pregnancy Brain Real?.txt",
    "Is ‘pregnancy brain’ real or just a myth? | Your Pregnancy Matters | UT Southwestern Medical Center.txt",
    "Memory and affective changes during the antepartum- A narrative review and integrative hypothesis- J.txt",
    "Pregnancy 'does cause memory loss' | Medical research | The Guardian.txt",
    "Pregnancy Brain — Forgetfulness During Pregnancy.txt",
    "Pregnancy brain- When it starts and what causes pregnancy brain fog | BabyCenter.txt",
    "Pregnancy does cause memory loss, study says - CNN.txt",
    "Textbook J.A. Russell, A.J. Douglas, R.J. Windle, C.D. Ingram - The Maternal Brain_ Neurobiological and Neuroendocrine Adaptation and Disorders in Pregnancy & Post Partum-Elsevier Science (2001).txt",
    "The effect of pregnancy on maternal cognition - PMC.txt",
    "This Is Your Brain on Motherhood - The New York Times.txt",
    "Working memory from pregnancy to postpartum.txt",
    "What Is Mom Brain and Is It Real?.txt",
    "Memory loss in Pregnancy- Myth or Fact? - International Forum for Wellbeing in Pregnancy.txt",
    "Memory and mood changes in pregnancy- a qualitative content analysis of women’s first-hand accounts.txt",
    "Is Mom Brain real? Understanding and coping with postpartum brain fog.txt",
    "Everyday Life Memory Deficits in Pregnant Women.txt",
    "Cognitive Function Decline in the Third Trimester.txt",
    "'Mommy brain' might be a good thing, new research suggests | CBC Radio.txt"
]

data = []
for file_path in txt_files:
    with open(file_path, "r") as file:
        text = file.read()
        data.append({"text": text})
        
df = pd.DataFrame(data)

# Cleaning the text
nltk.download('punkt')  # Download the 'punkt' tokenizer models for tokenization
df['tokens'] = df['text'].apply(word_tokenize) # Tokenize each text in the 'text' column into a list of words

nltk.download('stopwords')  # Download the stopwords corpus for English
stop_words = set(stopwords.words('english'))  # Load the English stopwords into a set

df['cleaned_text'] = df['tokens'].apply(lambda x: [word.lower() for word in x if (word.isalnum() and word.lower() not in stop_words)])

df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x))



# Function to generate response using fine-tuned GPT-2 model
def fine_tuned_generate_response(input_text):
    try:
        input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors='pt')
        output = fine_tuned_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=fine_tuned_tokenizer.eos_token_id, attention_mask=input_ids.ne(fine_tuned_tokenizer.eos_token_id))
        response = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"



# Streamlit app
context = []

try:
    st.title("Memory Support Chatbox for Pregnant Women")
    user_input = st.text_input("You:", "Enter your message here...", key="user_input")
    if st.button("Submit"):  # Use a button to submit the input
        context.append(user_input)
        input_text = " ".join(context[-3:])  # Consider last 3 messages as context
        reply_text = fine_tuned_generate_response(input_text)
        st.write(reply_text)
        st.text_area("Chatbot:", value=reply_text, height=200)
        st.experimental_rerun()  # Clear the input field after user submits
except Exception as e:
    st.error(f"An error occurred: {e}")



# Text Analysis
st.subheader("Text Analysis")

# Word Cloud
st.subheader("Word Cloud")

try:
    all_text = " ".join(df["cleaned_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
    plt.show()  # Move plt.show() here
except Exception as e:
    st.error(f"An error occurred: {e}")



