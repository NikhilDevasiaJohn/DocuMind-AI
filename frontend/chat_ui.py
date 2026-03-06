import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("DocuMind AI")
st.write("Ask questions about your documents")

query = st.text_input("Ask a question")

if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"question": query},
                timeout=30
            )
            response.raise_for_status()
            answer = response.json()["answer"]
            st.write("### Answer")
            st.write(answer)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The model may be taking too long to respond.")
        except Exception as e:
            st.error(f"An error occurred: {e}")