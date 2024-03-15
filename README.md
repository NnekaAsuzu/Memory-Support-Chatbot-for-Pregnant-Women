# Memory Support Chatbot for Pregnant Women

## Overview

This project aims to develop a Memory Support Chatbot for Pregnant Women using the GPT-2 language model. The chatbot will provide support and information to pregnant women experiencing cognitive memory issues during pregnancy.

## Dataset

The project uses text data related to pregnancy and memory issues. The dataset includes information about the types of cognitive memory issues reported by pregnant women.

## Methodology

The methodology for developing the Memory Support Chatbot includes:

1. **Data Cleaning and Preprocessing:** Cleaning and tokenizing the text data to prepare it for analysis and model training.
2. **Data Visualization:** Visualizing the types of cognitive memory issues and their frequencies reported by pregnant women.
3. **Model Integration:** Integrating the cleaned text data into the GPT-2 model to generate responses for the chatbot.
4. **Model Evaluation:** Evaluating the performance of the chatbot in providing relevant and helpful responses to user queries.

## Results

The Memory Support Chatbot demonstrates the ability to provide supportive and informative responses to pregnant women experiencing cognitive memory issues. The chatbot aims to improve user experience and provide valuable support during pregnancy.

## Future Improvements

Future improvements to the chatbot could include:

- Fine-tuning the GPT-2 model to improve response accuracy and relevance.
- Incorporating user feedback to continuously improve the chatbot's performance and usability.
- Expanding the chatbot's capabilities to provide additional support and information related to pregnancy and memory issues.

## Project Constraints

Issues/Constraints:

- **No Dataset Found:** The lack of an existing dataset required the use of text files instead, which required additional cleaning and preprocessing efforts.
- **Imperfect Responses:** Despite fine-tuning the GPT-2 model, responses may not always be perfect or tailored to specific needs.
- **Limited Resources:** Financial constraints prevented the use of paid open APIs, leading to the use of GPT-2 as an alternative solution.

## Deployment

The Memory Support Chatbot has been tested on both Gradio and Streamlit for deployment. Both platforms provide user-friendly interfaces for interacting with the chatbot.

- **Gradio:** Gradio offers a simple interface for deploying and testing machine learning models, including chatbots. The chatbot can be accessed through a web browser, allowing users to input their queries and receive responses from the GPT-2 model.
- **Streamlit:** Streamlit provides a platform for building and deploying data apps, including chatbots. The chatbot can be deployed as a web app, allowing users to interact with it through a user-friendly interface.

## Contributor(s)

Nneka Asuzu


## App Screenshots

### Gradio Interface

![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb1.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb2.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb3.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb4.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb5.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb6.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb7.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb8.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb9.png)
![Gradio Interface 1](/gradiowebsitescreenshots/gradioweb10.png)

### Streamlit Interface

![Streamlit Interface](/Streamlitewebsitescreenshot/streamlitappwebpic.png)


## Setup

To run the Memory Support Chatbot for Pregnant Women locally, follow these steps:

1. Clone the repository: `git clone https://github.com/NnekaAsuzu/Memory-Support-Chatbot-for-Pregnant-Women.git`
2. Navigate to the project directory: `cd Memory-Support-Chatbot-for-Pregnant-Women`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the chatbot with Gradio: `python gradio_app.py`
5. Run the chatbot with Streamlit: `streamlit run streamlit_app.py`

