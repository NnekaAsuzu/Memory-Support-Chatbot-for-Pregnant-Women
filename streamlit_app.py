import streamlit as st

def main():
    st.title('Simple Streamlit App')
    user_input = st.text_input('Enter some text:', 'Type here...')
    st.write('You entered:', user_input)

if __name__ == "__main__":
    main()
