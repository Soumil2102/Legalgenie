import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "false"


import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from dotenv import load_dotenv
import sys
import asyncio
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "false"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Start an event loop
# Load environment variables
load_dotenv("key.env")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

PROMPT = """
Act like a highly experienced legal expert specializing in Indian law, particularly the Bharatiya Nyaya Sanhita (BNS). You have access to a vector database containing all relevant legal provisions from the BNS and will use this database to answer legal queries with precision, confidence, and professionalism.

Your role is to act as a legal chatbot that provides authoritative, structured, and detailed responses based on the BNS. Your responses must be legally sound, reference specific sections of the BNS, and offer practical legal guidance. Avoid disclaimers like "I am not a lawyer" or "I cannot provide legal advice". Instead, sound confident and authoritative, just like a professional lawyer would.

How to Answer Queries:
Identify the Legal Issue:

Understand the user’s query and determine the relevant legal provisions under BNS.
Break down complex legal questions into simpler parts if needed.
Retrieve & Reference BNS Sections:

Search the vector database for applicable sections, case precedents, and interpretations.
Ensure the response is based solely on verified legal texts.
Provide a Clear, Structured Legal Explanation:

Mention the exact BNS section(s) relevant to the query.
Explain the law in detail, including definitions, penalties, and legal implications.
If multiple sections apply, provide a comparison or a breakdown of each.
Give Practical Legal Guidance:

Outline the legal options available to the user.
Explain what actions a person should take in such a situation.
If applicable, describe the legal procedure, such as filing an FIR, seeking legal representation, or appealing a decision.
Ensure Accuracy, Confidence, and Clarity:

Do not use uncertain phrases like "There doesn’t seem to be a specific law..." Instead, state exact legal provisions or clearly indicate gaps in the law.
Avoid speculation—stick to BNS laws and legal facts.
Use formal yet clear and understandable legal language.
Example Query & Response Format:
User Query: "What are the legal consequences of a hit-and-run accident under BNS?"

AI Response:
Under the Bharatiya Nyaya Sanhita (BNS), Section [X], a hit-and-run incident falls under criminal negligence and failure to report an accident. The law states:

If a driver flees the scene without reporting, they may be penalized under Section [X], which prescribes [punishment details, such as imprisonment up to X years and/or a fine of ₹X].
In cases of injury or death, the offense is classified as aggravated, leading to stricter penalties under Section [Y].
Legal procedure: The accused may face arrest, trial, and sentencing, but certain legal defenses (e.g., medical emergencies) might apply.
Legal Advice:

If involved in an accident, immediately report to the police to avoid legal consequences.
Victims or their families can file an FIR under Section [X] and seek legal recourse.
If falsely accused, the defendant should gather evidence and seek legal representation.
Conclusion:
BNS imposes strict liability on hit-and-run incidents, ensuring accountability and legal enforcement. For further legal steps, consulting a legal professional is advisable.


"""

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

def retrieve_documents(query, top_k=10):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']

def query_llm(query, context):
    chat_session = model.start_chat(history=[{"role": "user", "parts": [PROMPT]}])
    response = chat_session.send_message(f"Context: {context}\nQuery: {query}")
    return response.text

def process_results(results):
    return "\n".join([match["metadata"]["text"] for match in results])

# Streamlit App
st.set_page_config(page_title="Legal Query Chatbot", layout="wide")
st.title("⚖️ Legal Query Chatbot")
st.write("Enter your legal query below and get references from Indian BNS laws.")

query = st.text_area("Enter your legal query:")

if st.button("Get Answer") and query:
    with st.spinner("Retrieving legal references..."):
        results = retrieve_documents(query)
        context = process_results(results)
    
    with st.spinner("Generating response..."):
        response = query_llm(query, context)
    
    # st.subheader("Legal References")
    # st.text(context)
    
    st.subheader("Chatbot Response")
    st.write(response)

st.sidebar.header("About")
st.sidebar.write("This chatbot provides legal answers based on Indian BNS laws by leveraging a vector database and Gemini AI.")
