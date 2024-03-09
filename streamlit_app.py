import streamlit as st

# Chatbox interface
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
st.pyplot()
