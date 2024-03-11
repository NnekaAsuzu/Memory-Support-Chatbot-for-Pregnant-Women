# -*- coding: utf-8 -*-
"""Memory Support Chatbot for Pregnant Women using gradio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/106TJLki27ZyfG_WiU_M47O8SOyhS8Dpl

#Proposal Title: Memory Support Chatbot for Pregnant Women.

Project Summary:
The proposed project is to develop a Memory Support Chatbox for Pregnant Women using the GPT-2 language model. The chatbox aims to provide support, solutions, tips, and advice for pregnant women experiencing cognitive memory issues. The goal is to offer a personalized and accessible resource for pregnant women to manage their cognitive memory challenges effectively.

Proposed Project and Reasoning:
The project is proposed to address the lack of easily accessible and personalized support for pregnant women facing cognitive memory issues. By leveraging the capabilities of the GPT-2 language model, the chatbox can provide instant responses and guidance, complementing the advice of healthcare professionals.

*Dictionary-Based DataFrame creation method: This method reads the text files into a list of dictionaries and then creates a pandas DataFrame from these dictionaries using streamlit as the Gradio. Model is fine-tuned*

Install Required Libraries
"""

# Libraries
!pip install gradio
!pip install gradio --upgrade
!pip install pandas      # Pandas for data manipulation and analysis
!pip install matplotlib  # Matplotlib for data visualization
!pip install nltk        # NLTK for natural language processing tasks
!pip install wordcloud   # Wordcloud for generating word clouds
!pip install transformers # Transformers for accessing the GPT-2 model  #library for accessing the GPT-2 model

"""Import Libraries"""

import pandas as pd                  # Pandas for data manipulation and analysis
import nltk                          # NLTK for natural language processing tasks
from nltk.corpus import stopwords    # Stopwords from NLTK
from nltk.tokenize import word_tokenize   # Word tokenizer from NLTK
import gradio as gr                  #Import Gradio library for building the user interface
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

# # Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')    # GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')      # GPT-2 model
model.resize_token_embeddings(len(tokenizer))         # Resize token embeddings

"""#Data Collection and Preparation: Clean the dataset by removing special characters, digits, and unnecessary whitespace"""

# List of file paths for the TXT files
#Define the list of file paths for the TXT files
txt_files = [
    "/content/Cognition in Pregnancy- Perceptions and Performance, 2005-2006 - Dataset - B2FIND.txt",
    "/content/Frontiers | Cognitive disorder and associated factors among pregnant women attending antenatal servi.txt",
    "/content/Frustrated By Brain Fog? How Pregnancy Actually Alters Yo....txt",
    "/content/Is Pregnancy Brain Real?.txt",
    "/content/Is ‘pregnancy brain’ real or just a myth? | Your Pregnancy Matters | UT Southwestern Medical Center.txt",
    "/content/Memory and affective changes during the antepartum- A narrative review and integrative hypothesis- J.txt",
    "/content/Pregnancy 'does cause memory loss' | Medical research | The Guardian.txt",
    "/content/Pregnancy Brain — Forgetfulness During Pregnancy.txt",
    "/content/Pregnancy brain- When it starts and what causes pregnancy brain fog | BabyCenter.txt",
    "/content/Pregnancy does cause memory loss, study says - CNN.txt",
    "/content/Textbook J.A. Russell, A.J. Douglas, R.J. Windle, C.D. Ingram - The Maternal Brain_ Neurobiological and Neuroendocrine Adaptation and Disorders in Pregnancy & Post Partum-Elsevier Science (2001).txt",
    "/content/The effect of pregnancy on maternal cognition - PMC.txt",
    "/content/This Is Your Brain on Motherhood - The New York Times.txt",
    "/content/Working memory from pregnancy to postpartum.txt",
    "/content/What Is Mom Brain and Is It Real?.txt",
    "/content/Memory loss in Pregnancy- Myth or Fact? - International Forum for Wellbeing in Pregnancy.txt",
    "/content/Memory and mood changes in pregnancy- a qualitative content analysis of women’s first-hand accounts.txt",
    "/content/Is Mom Brain real? Understanding and coping with postpartum brain fog.txt",
    "/content/Everyday Life Memory Deficits in Pregnant Women.txt",
    "/content/Cognitive Function Decline in the Third Trimester.txt",
    "/content/'Mommy brain' might be a good thing, new research suggests | CBC Radio.txt"
]

#Load and read the text files into a DataFrame
data = []
for file_path in txt_files:
    with open(file_path, "r") as file:
        text = file.read()
        data.append({"text": text})

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

"""##Data Cleaning and Manipulation"""

# Tokenize the text
nltk.download('punkt')  # Download the 'punkt' tokenizer models for tokenization
df['tokens'] = df['text'].apply(word_tokenize) # Tokenize each text in the 'text' column into a list of words

# Remove stopwords and special characters
nltk.download('stopwords')  # Download the stopwords corpus for English
stop_words = set(stopwords.words('english'))  # Load the English stopwords into a set

# Apply a lambda function to each tokenized list in the 'tokens' column
# This function converts each word to lowercase, removes non-alphanumeric characters, and filters out stopwords
df['cleaned_text'] = df['tokens'].apply(lambda x: [word.lower() for word in x if (word.isalnum() and word.lower() not in stop_words)])

# Join tokens back into sentences
# Apply a lambda function to join the cleaned tokens back into a single string
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x))

# Display the cleaned text
print(df['cleaned_text'])

"""##Exploratory Data Analysis and Visualization

*  Perform basic statistics on the dataset, such as word count, average length of articles (Performing basic statistics on a dataset, such as word count and average article length, helps understand the dataset's content complexity and typical article length, aiding further analysis and processing)
"""

# Word count
df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

# Average length of articles
average_length = df['word_count'].mean()

# Minimum and maximum word count
min_word_count = df['word_count'].min()
max_word_count = df['word_count'].max()

print(f"Average length of articles: {average_length:.2f} words")
print(f"Minimum word count: {min_word_count} words")
print(f"Maximum word count: {max_word_count} words")

# Display the cleaned text and word count
print(df[['cleaned_text', 'word_count']])

# Visualize the distribution of word counts
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=20, color='brown', edgecolor='black')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Distribution of Word Counts')
plt.show()

"""*  Word Cloud (Display the most common words or phrases, providing a  visual representation of the main themes and topics discussed in the articles)"""

# Concatenate all cleaned text into a single string
all_text = " ".join(df["cleaned_text"])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Word Cloud of Cognitive Memory Issues')
plt.show()

"""

*  Frequency of Cognitive Memory Issues (Describes the frequency of different types of cognitive memory issues reported in the text data. It shows the number of occurrences of each type of issue, such as memory loss, difficulty concentrating, forgetfulness, brain fog, and others, providing insights into the prevalence of these issues in the dataset)

"""

def cleaned_text(text):
    # Your cleaning logic here
    cleaned_text = text.lower()  # Example: convert text to lowercase
    return cleaned_text

# Apply cleaning function to 'text' column and store in 'cleaned_text' column
df["cleaned_text"] = df["text"].apply(cleaned_text)

# Define types of cognitive memory issues
types_of_issues = ['Memory Loss', 'Difficulty Concentrating', 'Forgetfulness', 'Brain Fog', 'Others']

# Initialize frequencies dictionary
frequencies = {issue: 0 for issue in types_of_issues}

# Count frequencies of each type of issue
for text in df["cleaned_text"]:
    for issue in types_of_issues:
        if issue.lower() in text:  # Example: use lowercase for comparison
            frequencies[issue] += 1

# Convert frequencies to DataFrame for plotting
df_frequencies = pd.DataFrame(list(frequencies.items()), columns=['Types of cognitive memory issues', 'Frequency'])

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_frequencies['Types of cognitive memory issues'], df_frequencies['Frequency'], color='skyblue')
plt.xlabel('Types of cognitive memory issues')
plt.ylabel('Frequency')
plt.title('Frequency of Cognitive Memory Issues')
plt.xticks(rotation=45)
plt.show()

# Display the cleaned text
print(df['cleaned_text'])

"""**Text Mining and NLP Analysis**
Goal is to extract key terms and topics related to cognitive memory issues during pregnancy, providing insights into the content and themes of the articles. This analysis helps identify patterns, trends, and prevalent topics in the literature, which can inform the development of the chatbox and enhance its ability to provide relevant and personalized support to pregnant women.
"""

# Text Mining and NLP
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Latent Dirichlet Allocation (LDA) for Topic Modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Extract key terms and topics
terms = tfidf_vectorizer.get_feature_names_out()
topics = [[terms[i] for i in topic.argsort()[:-6:-1]] for topic in lda.components_]

# Print topics
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {', '.join(topic)}")

"""**Sentiment Analysis and Comparison and Contrast:** This analysis assigns a sentiment score to each article, indicating its overall sentiment (positive, neutral, or negative). The sentiment scores can help understand the general tone and attitude of the articles towards cognitive memory issues during pregnancy."""

# Sentiment Analysis
# Calculate the sentiment score for each cleaned text and store it in a new column
df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Explanation of sentiment scores:
# -1 indicates extremely negative sentiment.
# 0 indicates neutral sentiment.
# 1 indicates extremely positive sentiment.

# Comparison and Contrast
# For simplicity, let's just plot the sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=20, color='green')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis of Articles')
plt.show()

"""#Comparing Baseline GPT-2 Model Responses Using Raw and Cleaned Text Before Fine-Tuning

##*For the baseline model with raw text:*
"""

# Baseline GPT-2 Model Response before cleaning
def baseline_generate_response(raw_text):
    # Tokenize the raw text
    input_ids = tokenizer(raw_text, return_tensors='pt')['input_ids']
    # Generate output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
raw_text1 = "How does pregnancy affect memory?"
raw_text2 = "What are the effects of pregnancy on cognitive function?"
baseline_response_raw1 = baseline_generate_response(raw_text1)
baseline_response_raw2 = baseline_generate_response(raw_text2)
print("Baseline Response (Raw Text 1):", baseline_response_raw1.rstrip('!'))
print("Baseline Response (Raw Text 2):", baseline_response_raw2.rstrip('!'))

"""##*For the baseline model with cleaned text:*"""

# Baseline GPT-2 Model Response #Generate responses using the GPT-2 model
def baseline_generate_response(cleaned_text):
    # Tokenize the cleaned text
    input_ids = tokenizer.encode(cleaned_text, return_tensors='pt')
    # Generate output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
cleaned_text1 = "How does pregnancy affect memory?"
cleaned_text2 = "What are the effects of pregnancy on cognitive function?"
baseline_response1 = baseline_generate_response(cleaned_text1)
baseline_response2 = baseline_generate_response(cleaned_text2)
print("Baseline Response 1:", baseline_response1)
print("Baseline Response 2:", baseline_response2)

"""#Fine-Tuning the GPT-2 Model

Tokenization and Padding for Fine-Tuning GPT-2 Model
"""

# Define the maximum sequence length
max_length = 512

# Tokenize the cleaned text and truncate to max_length
df['tokenized_text'] = df['cleaned_text'].apply(lambda x: tokenizer.encode(x, max_length=max_length, truncation=True, return_tensors='pt'))

# Get the padding value from the tokenizer or use a default value
padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

# Pad or truncate the tokenized sequences to the maximum length
padded_sequences = pad_sequence([seq.squeeze(0) for seq in df['tokenized_text']], batch_first=True, padding_value=padding_value)

# Concatenate the padded sequences to match the expected size
input_ids = torch.cat(tuple(padded_sequences), dim=0)
labels = input_ids.clone()

"""Fine-Tuning: Ready to fine-tune the GPT-2 model"""

#Check if 'input_ids' and 'labels' are defined
if 'input_ids' in locals() and 'labels' in locals():
    print("input_ids and labels are defined.")
else:
    print("input_ids and labels are not defined.")

# Define hyperparameters
num_epochs = 15   # Increased to 15 epochs
learning_rate = 5e-5  # Adjusted learning rate
weight_decay = 0.01   # Adjusted weight decay
warmup_steps = 500    # Adjusted warmup steps
max_seq_length = 1024  # Maximum sequence length

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_epochs * len(df)
)

"""Manual Training Loop Method: In this method, the training loop is implemented manually, without using a custom trainer class."""

# Fine-tune the GPT-2 model
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

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2_model')

# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_model')

"""Comparison of Baseline and Fine-Tuned GPT-2 Model Responses

"""

# Baseline GPT-2 Model Response
def baseline_generate_response(input_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Generate output
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Fine-Tuned GPT-2 Model Response
def fine_tuned_generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = fine_tuned_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
input_text = "How does pregnancy affect memory?"

baseline_response = baseline_generate_response(input_text)
fine_tuned_response = fine_tuned_generate_response(input_text)

print("Baseline Response:", baseline_response)
print("Fine-Tuned Response:", fine_tuned_response)

# Examples usage

# Example 1
input_text = "Are there any foods or supplements that can help with memory during pregnancy??"
baseline_response = baseline_generate_response(input_text)
fine_tuned_response = fine_tuned_generate_response(input_text)
print("Baseline Response:", baseline_response)
print("Fine-Tuned Response:", fine_tuned_response)


# Example 2
input_text = "Can pregnancy brain fog affect my ability to work or perform daily tasks?"
baseline_response = baseline_generate_response(input_text)
fine_tuned_response = fine_tuned_generate_response(input_text)
print("Baseline Response:", baseline_response)
print("Fine-Tuned Response:", fine_tuned_response)


# Example 3
input_text = "Should I be concerned if I'm experiencing more forgetfulness than usual during pregnancy?"
baseline_response = baseline_generate_response(input_text)
fine_tuned_response = fine_tuned_generate_response(input_text)
print("Baseline Response:", baseline_response)
print("Fine-Tuned Response:", fine_tuned_response)

# Compare the responses
print("Are the responses the same?")
print(baseline_response == fine_tuned_response)

"""#Deployment: Chatbox Development: Define the Gradio app interface and functionality





"""

# Function for the chatbox
def chatbox(message_chatbot):
    try:
        chatbot_response = fine_tuned_generate_response(message_chatbot)
        return chatbot_response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbox,
    inputs="text",
    outputs="text",
    title="Memory Support Chatbox for Pregnant Women",
    description="Ask any question about cognitive memory issues during pregnancy."
)
iface.launch()

"""#Model Evaluation"""

#Generate sample responses from the model to ensure it provides relevant and coherent advice

# Sample prompts
sample_prompts = [
    "What causes pregnancy brain fog?",
    "How does pregnancy affect the brain?",
    "How can I improve my memory during pregnancy?",
    "Can pregnancy brain fog affect my ability to work or perform daily tasks?",
]

# Generate responses
for prompt in sample_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Response:", response)

"""##Visualization of Model Responses to Sample Prompts

Can help one  understand the nature of the responses generated by the language model and
identify common patterns or topics across the responses
"""

#Sample prompts
sample_prompts = [
    "What causes pregnancy brain fog?",
    "How does pregnancy affect the brain?",
    "How can I improve my memory during pregnancy?",
    "Can pregnancy brain fog affect my ability to work or perform daily tasks?",
]

# Generate responses from the model
data = []
for prompt in sample_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.ne(tokenizer.eos_token_id))
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    data.append({"prompt": prompt, "response": response})

# Create a DataFrame from the generated responses
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Use the generated responses for visualization
# For example, you can create a pie chart based on the responses
# Pie chart
plt.figure(figsize=(8, 8))
plt.pie([len(response.split()) for response in df['response']], labels=df['prompt'], autopct='%1.1f%%', startangle=90)
plt.title('Distribution of response lengths for sample prompts')
plt.axis('equal')
plt.show()