# -*- coding: utf-8 -*-

import pandas as pd                  # Pandas for data manipulation and analysis            
import nltk                          # NLTK for natural language processing tasks
from nltk.corpus import stopwords    # Stopwords from NLTK
from nltk.tokenize import word_tokenize   # Word tokenizer from NLTK
import streamlit as st               # Streamlit for creating interactive web apps
import matplotlib.pyplot as plt     # Matplotlib for data visualization
from wordcloud import WordCloud     # Wordcloud for generating word clouds
from transformers import GPT2Tokenizer, GPT2LMHeadModel   # GPT-2 model from Transformers
import torch                         # PyTorch for deep learning tasks
from torch.utils.data import DataLoader, Dataset,TensorDataset   # DataLoader and Dataset for handling data
from transformers import GPT2Config, GPT2LMHeadModel, AdamW     # AdamW optimizer for GPT-2 model
from transformers import AdamW, get_scheduler                    # Scheduler for optimizer
from torch.nn.utils.rnn import pad_sequence                      # Padding sequences for model input
from nltk.sentiment import SentimentIntensityAnalyzer            # Sentiment analysis from NLTK
from sklearn.feature_extraction.text import TfidfVectorizer      # TF-IDF vectorizer
from sklearn.decomposition import LatentDirichletAllocation      # LDA for topic modeling
nltk.download('vader_lexicon')  # Download the VADER lexicon for sentiment analysis
sia = SentimentIntensityAnalyzer()  # Initialize the SentimentIntensityAnalyzer


import nltk

def setup_nltk():
    try:
        # Download NLTK resources if not already downloaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')

setup_nltk()


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')   # GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')  # GPT-2 model
model.resize_token_embeddings(len(tokenizer))   # Resize token embeddings

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


print(df)


nltk.download('punkt')  # Download the 'punkt' tokenizer models for tokenization
df['tokens'] = df['text'].apply(word_tokenize) # Tokenize each text in the 'text' column into a list of words


nltk.download('stopwords')  # Download the stopwords corpus for English
stop_words = set(stopwords.words('english'))  # Load the English stopwords into a set


df['cleaned_text'] = df['tokens'].apply(lambda x: [word.lower() for word in x if (word.isalnum() and word.lower() not in stop_words)])


df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x))


print(df['cleaned_text'])


df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))


average_length = df['word_count'].mean()


min_word_count = df['word_count'].min()
max_word_count = df['word_count'].max()

print(f"Average length of articles: {average_length:.2f} words")
print(f"Minimum word count: {min_word_count} words")
print(f"Maximum word count: {max_word_count} words")


print(df[['cleaned_text', 'word_count']])

plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=20, color='brown', edgecolor='black')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Counts')
plt.show()


all_text = " ".join(df["cleaned_text"])


wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Word Cloud of Cognitive Memory Issues')
plt.show()



def cleaned_text(text):
    
    cleaned_text = text.lower()  # Example: convert text to lowercase
    return cleaned_text


df["cleaned_text"] = df["text"].apply(cleaned_text)


types_of_issues = ['Memory Loss', 'Difficulty Concentrating', 'Forgetfulness', 'Brain Fog', 'Others']


frequencies = {issue: 0 for issue in types_of_issues}


for text in df["cleaned_text"]:
    for issue in types_of_issues:
        if issue.lower() in text:  # Example: use lowercase for comparison
            frequencies[issue] += 1


df_frequencies = pd.DataFrame(list(frequencies.items()), columns=['Types of cognitive memory issues', 'Frequency'])


plt.figure(figsize=(10, 6))
plt.bar(df_frequencies['Types of cognitive memory issues'], df_frequencies['Frequency'], color='skyblue')
plt.xlabel('Types of cognitive memory issues')
plt.ylabel('Frequency')
plt.title('Frequency of Cognitive Memory Issues')
plt.xticks(rotation=45)
plt.show()


print(df['cleaned_text'])




tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)


terms = tfidf_vectorizer.get_feature_names_out()
topics = [[terms[i] for i in topic.argsort()[:-6:-1]] for topic in lda.components_]


for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {', '.join(topic)}")



df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=20, color='green')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis of Articles')
plt.show()





def baseline_generate_response(raw_text):
    # Tokenize the raw text
    input_ids = tokenizer(raw_text, return_tensors='pt')['input_ids']
    # Generate output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


raw_text1 = "How does pregnancy affect memory?"
raw_text2 = "What are the effects of pregnancy on cognitive function?"
baseline_response_raw1 = baseline_generate_response(raw_text1)
baseline_response_raw2 = baseline_generate_response(raw_text2)
print("Baseline Response (Raw Text 1):", baseline_response_raw1.rstrip('!'))
print("Baseline Response (Raw Text 2):", baseline_response_raw2.rstrip('!'))




def baseline_generate_response(cleaned_text):
    input_ids = tokenizer.encode(cleaned_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


cleaned_text1 = "How does pregnancy affect memory?"
cleaned_text2 = "What are the effects of pregnancy on cognitive function?"
baseline_response1 = baseline_generate_response(cleaned_text1)
baseline_response2 = baseline_generate_response(cleaned_text2)
print("Baseline Response 1:", baseline_response1)
print("Baseline Response 2:", baseline_response2)




max_length = 512


df['tokenized_text'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x[:max_length], return_tensors='pt'))


padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

padded_sequences = pad_sequence([seq.squeeze(0)[:max_length] for seq in df['tokenized_text']], batch_first=True, padding_value=padding_value)


input_ids = torch.cat(tuple(padded_sequences), dim=0)
labels = input_ids.clone()


if 'input_ids' in locals() and 'labels' in locals():
    print("input_ids and labels are defined.")
else:
    print("input_ids and labels are not defined.")


num_epochs = 3
learning_rate = 5e-5  # Adjusted learning rate
weight_decay = 0.01   # Adjusted weight decay
warmup_steps = 500    # Adjusted warmup steps
max_seq_length = 1024  # Maximum sequence length

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(df) * num_epochs)


model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for text in df['cleaned_text']:
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    average_loss = total_loss / len(df)
    print(f"Epoch {epoch+1}: Average Loss = {average_loss}")


model.save_pretrained('fine_tuned_gpt2_model')


fine_tuned_model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_model')



def baseline_generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Generate output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def fine_tuned_generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = fine_tuned_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


input_text = "How does pregnancy affect memory?"

baseline_response = baseline_generate_response(input_text)
fine_tuned_response = fine_tuned_generate_response(input_text)

print("Baseline Response:", baseline_response)
print("Fine-Tuned Response:", fine_tuned_response)

print("Are the responses the same?")
print(baseline_response == fine_tuned_response)

try:
    st.title("Memory Support Chatbox for Pregnant Women")
    user_input = st.text_input("You:", "Enter your message here...")
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        reply_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        st.text_area("Chatbot:", value=reply_text, height=200)
except Exception as e:
    st.error(f"An error occurred: {e}")


st.subheader("Text Analysis")

st.subheader("Word Cloud")
all_text = " ".join(df["cleaned_text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot()




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
    print("Response:", response)



sample_prompts = [
    "What causes pregnancy brain fog?",
    "How does pregnancy affect the brain?",
    "How can I improve my memory during pregnancy?",
    "Can pregnancy brain fog affect my ability to work or perform daily tasks?",
]


data = []
for prompt in sample_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    data.append({"prompt": prompt, "response": response})



df = pd.DataFrame(data)


print(df)



plt.figure(figsize=(8, 8))
plt.pie([len(response.split()) for response in df['response']], labels=df['prompt'], autopct='%1.1f%%', startangle=90)
plt.title('Distribution of response lengths for sample prompts')
plt.axis('equal')
plt.show()
