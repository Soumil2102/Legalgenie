import os
import base64
import asyncio
import sys
from pathlib import Path
from docx import Document
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Fix for asyncio in Streamlit
os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "false"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))  # Start an event loop

# Load environment variables
load_dotenv("key.env")

# Access API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Ensure API keys are loaded
if not gemini_api_key or not pinecone_api_key or not pinecone_index_name:
    raise ValueError("API keys not found! Make sure key.env file is correctly set up.")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

# Configuration for the Gemini model
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

# Define prompts
LEGAL_QUERY_PROMPT = """
Act like a highly experienced legal expert specializing in Indian law, particularly the Bharatiya Nyaya Sanhita (BNS). You have access to a Pinecone vector database containing all relevant legal provisions from the BNS and will use this database to answer legal queries with precision, confidence, and professionalism.

Your role is to act as a legal chatbot that provides authoritative, structured, and detailed responses based on the BNS. Your responses must be legally sound, reference specific sections of the BNS, and offer practical legal guidance. Avoid disclaimers like "I am not a lawyer" or "I cannot provide legal advice". Instead, sound confident and authoritative, just like a professional lawyer would.

How to Answer Queries:
Identify the Legal Issue:

Understand the user's query and determine the relevant legal provisions under BNS.
Break down complex legal questions into simpler parts if needed.
Retrieve & Reference BNS Sections:

Search the Pinecone vector database for applicable sections, case precedents, and interpretations.
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

Do not use uncertain phrases like "There doesn't seem to be a specific law..." Instead, state exact legal provisions or clearly indicate gaps in the law.
Avoid speculation—stick to BNS laws and legal facts.
Use formal yet clear and understandable legal language.
"""

# Dictionary of document templates for different legal document types
DOCUMENT_TEMPLATES = {
    "divorce_petition": """IN THE FAMILY COURT AT MUMBAI
PETITION No. / 2024

IN THE MATTER OF

NAME : 
AGE : 
OCCUPATION : 
ADDRESS : 
Mobile No.
Email ID ….PETITIONER NO. 1

AND

NAME : 
AGE : 
OCCUPATION : 
ADDRESS : 
Mobile No.
Email ID ….PETITIONER NO. 2

A Petition For divorce by mutual consent U/s
(SPECIFY UNDER WHICH ACT, whether)
U/S 13B Of Hindu Marriage Act
Or
U/S 28 Of Special Marriage Act
Or
U/S 10 A Of Divorce Act

The petitioner above named submits this petition praying to state as follows;

1. That the petitioners were married to each other at …......................... on dated.............................. according to the................................rites and customs/ceremonies.
Or before the Marriage Registrar ….............(Name of City/Town)

2. That the petitioner no. 1 before marriage was ….............and petitioner no. 2 was …................. 
[State the pre marital status of the parties whether bachelor/ spinster/ divorcee/ widow/ widower.
Mention the maiden name of the wife.
Mention the religion and domicile of the parties
Clearly mention the date since when the parties are staying separately]

3. [State the number of children. Their names and age/ date of birth and custody.]

4. [State the details about pending litigation. Under which section, Act, case number and court. Next date fixed before the competant court.]

5. [State the details about joint immovable property, if any.]

6. CONSENT TERMS
[The consent terms must include what the parties decided about
- The permanent alimony,
- Custody and access of children,
- Division of property/ execution of any regd document in respect of immovable property Exchange of articles/jwellery/utencils etc,
- Withdrawal of pending litigations, and
- Any other term to which the parties are consenting]

7. That the petitioners due hereby declare and confirm that this petition preferred by them is not collusive.

8. That there is no coercion, force, fraud, undue influence, misrepresentation etc. in filing the present petition, and our consent is free.

9. That there is no collusion or connivance between the parties in filing this petition.

10. That this Court has jurisdiction to try and decide this petition as
[Mention clearly how this court has jurisdiction.
- Whether the marriage was solemnized at Mumbai.
- That the parties lastly stayed together at Mumbai.
- The wife is staying at Mumbai.
- Any other reason supported by document.]

11. That the court fee of Rs. 100 is affixed.

12. The petitioners will rely upon the documents, a list whereof is annexed herewith.

13. The petitioners pray that;
a) This Hon'ble court be pleased to dissolve the marriage between the petitioners, solemnized on ….............. by the decree of divorce by mutual consent under section ….............................
b) Such other and further relief's as this Hon'ble Court may deem fit and proper in the nature and circumstances of the case;

VERIFICATION

I …............................. age :....................... years, residing at ….......... the petitioner no. 1 do hereby solemnly declare that what is stated in the foregoing paragraphs of the petition is true to best of my own knowledge and belief save and except for the legal submission.

Solemnly Declared at ….........
On this ….....................(Date)
Signature of the petitioner no. 1

Advocate

I …............................. age :....................... years, residing at …......... ..the petitioner no. 2 do hereby solemnly declare that what is stated in the foregoing paragraphs of the petition is true to best of my own knowledge and belief save and except for the legal submission.

Solemnly Declared at ….........
On this ….....................(Date)
Signature of the petitioner no. 2

Advocate

Documents to be attached:
- ID proof of both the parties (Copy of Pan Card/ Driving license /Adhar Card / Election Card/ Passport).
- Marriage proof (Marriage Registration Certificate/ Invitation Card/ Marriage Photograph/ Affidavit of blood relative) (Minimum two documents mandatory).
- Residential proof (Passport/ Adhar Card/ Election Card/ any other permissable document).

Additional Documents if required:
- Birth Certificate of minor child.
- Registered document for transfer of property.
- Copy of receipt if articles, jwellery, or utencils are exchanged.""",

    "rental_agreement": """RENT AGREEMENT

THIS RENT AGREEMENT is made on this __ day of ______, 20__ at _______ BETWEEN ________________ S/o, D/o, W/o __________________, Residing at ___________________ (hereinafter referred to as the "LESSOR") of the ONE PART.

AND

_________________ S/o, D/o, W/o __________________, Residing at ___________________ (hereinafter referred to as the "LESSEE") of the OTHER PART.

The terms "LESSOR" and "LESSEE" shall mean and include their respective heirs, successors, assigns, representatives, etc.

WHEREAS the LESSOR is the absolute owner of the residential/commercial premises bearing No._____________ consisting of ______ situated at _____________ (hereinafter referred to as the "SCHEDULE PREMISES").

AND WHEREAS the LESSEE has approached the LESSOR and requested to let out the SCHEDULE PREMISES for a period of _____ months/years commencing from __________ for residential/commercial purpose, and the LESSOR has agreed to the same on the following terms and conditions.

NOW THIS RENT AGREEMENT WITNESSETH AS FOLLOWS:

1. RENT:
   The LESSEE shall pay to the LESSOR rent at the rate of Rs.______ (Rupees ______________ only) per month, payable in advance on or before the ___ day of each English Calendar month.

2. DURATION:
   This Agreement shall be for a period of ____ months/years commencing from __________ and ending on __________. This Agreement may be renewed for another term by mutual consent of both the parties on such terms and conditions as may be agreed upon by them.

3. SECURITY DEPOSIT:
   The LESSEE has paid to the LESSOR a sum of Rs.______ (Rupees ______________ only) as interest-free refundable security deposit, which shall be refunded by the LESSOR to the LESSEE at the time of vacating the SCHEDULE PREMISES, after deducting therefrom any arrears of rent, electricity, water charges or any other charges payable by the LESSEE under this Agreement or any damages caused to the SCHEDULE PREMISES by the LESSEE.

4. PAYMENT OF ELECTRICITY AND WATER CHARGES:
   The LESSEE shall pay the electricity and water charges as per the respective meter readings on the due dates to the concerned authorities directly.

5. MAINTENANCE CHARGES:
   The LESSEE shall pay the monthly maintenance charges of Rs.______ (Rupees ______________ only) to the [Society/Building/Corporation] directly.

6. USE OF PREMISES:
   The LESSEE shall use the SCHEDULE PREMISES for residential/commercial purpose only and shall not use it for any illegal or immoral purposes. The LESSEE shall not cause any nuisance or annoyance to the neighbors.

7. REPAIRS AND MAINTENANCE:
   The LESSEE shall keep the SCHEDULE PREMISES in good and tenantable condition and shall be responsible for minor repairs. Any major structural repairs shall be the responsibility of the LESSOR.

8. SUB-LETTING:
   The LESSEE shall not sub-let, sub-lease, or assign the SCHEDULE PREMISES or any part thereof to any third party under any circumstances without the prior written consent of the LESSOR.

9. INSPECTION:
   The LESSOR or his authorized representative shall have the right to inspect the SCHEDULE PREMISES after giving reasonable notice to the LESSEE.

10. TERMINATION:
    Either party may terminate this Agreement by giving ____ months' notice in writing to the other party.

11. RETURN OF SCHEDULE PREMISES:
    On the expiry of the term of this Agreement or its earlier termination, the LESSEE shall peacefully and quietly deliver vacant possession of the SCHEDULE PREMISES to the LESSOR in the same condition as it was at the time of taking possession, subject to natural wear and tear.

12. JURISDICTION:
    Any dispute arising out of this Agreement shall be subject to the jurisdiction of the Courts in ___________.

IN WITNESS WHEREOF the parties hereto have set their hands to this Rent Agreement on the day, month and year first above written.

LESSOR                                      LESSEE

_________________                          _________________
(Signature)                                (Signature)

WITNESSES:

1. ________________                        2. ________________
   (Signature)                                (Signature)
   Name:                                      Name:
   Address:                                   Address:"""
}

# Function definitions
def retrieve_documents(query, top_k=10):
    """Retrieve relevant documents from Pinecone vector database."""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']

def process_results(results):
    """Process the results from Pinecone into a single string."""
    return "\n".join([match["metadata"]["text"] for match in results])

def query_llm(query, context):
    """Query the Gemini model with the given query and context."""
    chat_session = model.start_chat(history=[{"role": "user", "parts": [LEGAL_QUERY_PROMPT]}])
    response = chat_session.send_message(f"Context: {context}\nQuery: {query}")
    return response.text

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def detect_document_type(file_path):
    """Detect the type of legal document."""
    file = upload_to_gemini(file_path, mime_type="application/pdf")
    
    document_type_prompt = """
    Analyze the uploaded document and identify what type of legal document it is. 
    Consider common Indian legal documents such as:
    - Divorce petition
    - Rental/lease agreement
    - Will/testament
    - Power of attorney
    - Sale deed
    - Employment contract
    - Partnership agreement
    - Loan agreement
    - Company incorporation documents
    - Consumer complaint
    - Criminal/civil case petition
    
    Respond with ONLY the document type in a single word or short phrase. If uncertain, respond with "other".
    """
    
    chat_session = model.start_chat()
    response = chat_session.send_message([file, document_type_prompt])
    
    # Clean up response to get just the document type
    doc_type = response.text.strip().lower()
    
    # Map detected document type to our template types
    if "divorce" in doc_type or "mutual consent" in doc_type:
        return "divorce_petition"
    elif "rent" in doc_type or "lease" in doc_type or "tenancy" in doc_type:
        return "rental_agreement"
    else:
        return "general"  # Default for documents without specific templates

def get_document_query_terms(doc_type):
    """Get the appropriate query terms based on document type."""
    query_terms = {
        "divorce_petition": "divorce petition mutual consent Indian law family court",
        "rental_agreement": "rental agreement lease tenancy Indian law property",
        "general": "Indian law legal document contract"
    }
    return query_terms.get(doc_type, query_terms["general"])

def process_document(file_path):
    """Process the document using the Gemini model and Pinecone for legal information."""
    # First detect document type
    doc_type = detect_document_type(file_path)
    
    # Upload file to Gemini
    files = [upload_to_gemini(file_path, mime_type="application/pdf")]
    
    # Get appropriate query terms for the document type
    document_query = get_document_query_terms(doc_type)
    
    # Retrieve relevant legal context
    legal_context_results = retrieve_documents(document_query, top_k=15)
    legal_context = process_results(legal_context_results)
    
    # Get template if available, otherwise use general analysis
    template = DOCUMENT_TEMPLATES.get(doc_type, "")
    template_instruction = f"Based on the document's contents, generate a draft following EXACTLY this template format:\n\n{template}" if template else "Generate a legally compliant draft based on the document's contents and current Indian legal standards."
    
    # Build document prompt based on document type
    document_prompt = f"""
    You are a highly skilled legal assistant specializing in Indian law. Using the uploaded PDF document as your only input, perform the following tasks:
     Analyze this document STRICTLY against Bharatiya Nyaya Sanhita (BNS) provisions:

    === BNS LEGAL CONTEXT ===
    {legal_context}

    Perform this analysis:
    1. Identify which BNS sections apply to this document
    2. Flag any clauses contradicting BNS provisions
    3. Suggest BNS-compliant alternatives
    
    ### **1. Document Summary:**
    - Summarize the document in one concise paragraph.
    - Focus on identifying the key parties involved, the legal grounds or purpose of the document, and any critical details.

    ### **2. Discrepancy Detection:**
    - Analyze the document for potential legal issues such as:
      - Missing mandatory clauses as per Indian law.
      - Incorrect or outdated statutory references.
      - Contradictory statements or procedural inconsistencies.
    - Provide a **bullet-point list of discrepancies**, citing specific Indian laws or judicial precedents that support your findings.
    - Suggest appropriate corrections based on current Indian legal practices.

    ### **3. Draft Generation:**
    {template_instruction}
    
    ### **4. Legal Verification:**
    Use the following information from legal databases to verify the legal compliance of the document:
    
    {legal_context}
    
    ### **5. Identify Incorrect Clauses:**  
    - Review the document thoroughly and list **any legally incorrect, outdated, or non-compliant clauses** based on **Indian laws and relevant regulatory guidelines**.  
    - Highlight provisions that **contradict Indian judicial precedents** or contain **ambiguous wording that may lead to legal disputes**.  
    - For each incorrect clause, provide a detailed explanation of why it is incorrect and cite relevant laws or precedents.

    ### **6. Provide Corrected Clauses:**  
    - Suggest legally accurate replacements for the incorrect clauses.  
    - Ensure that the revised clauses align with **Indian legal standards, case laws, and contract enforceability principles**.  
    - Maintain clarity, precision, and compliance with standard legal drafting conventions used in **Indian agreements**.  

    ### **7. Identify Missing Clauses (if any):**  
    - Check if the agreement is missing any **mandatory clauses** required under Indian law.  
    - Suggest additional clauses that enhance **legal protection, risk mitigation, and enforceability**.  
    - Provide a detailed explanation of why each missing clause is necessary and how it should be drafted.

    Provide your output in the following format:
    Summary: [your summary]
    
    Discrepancies: [your list of discrepancies]
    
    Incorrect Clauses: [your analysis]
    
    Corrected Clauses: [your suggestions]
    
    Missing Clauses: [your analysis]
    
    Draft: [your generated draft]
    """

    chat_session = model.start_chat(history=[{"role": "user", "parts": [files[0], document_prompt]}])
    response = chat_session.send_message("Generate legal analysis and draft based on the provided document.")
    return response.text, doc_type

def create_word_document(text, doc_type, filename=None):
    """Create a Word document from the given text with appropriate filename."""
    if filename is None:
        type_names = {
            "divorce_petition": "divorce_petition",
            "rental_agreement": "rental_agreement",
            "general": "legal_document"
        }
        filename = f"{type_names.get(doc_type, 'legal_document')}.docx"
        
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    return filename

def extract_section(result, section_name):
    """Extract a specific section from the result text."""
    try:
        parts = result.split(f"{section_name}:")
        if len(parts) > 1:
            # Find the next section marker or end of text
            section_text = parts[1].strip()
            for next_section in ["Summary:", "Discrepancies:", "Incorrect Clauses:", "Corrected Clauses:", "Missing Clauses:", "Draft:"]:
                if next_section in section_text and next_section != f"{section_name}:":
                    section_text = section_text.split(next_section)[0].strip()
            return section_text
        return f"{section_name} section not found in the response."
    except Exception as e:
        return f"Error extracting {section_name}: {str(e)}"

# Streamlit App Configuration
st.set_page_config(
    page_title="Legal Assistant AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Enhanced Black-Blue Theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2c89c5;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    .stFileUploader>div>div>div>button {
        background-color: #1F77B4;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f29;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F77B4;
        border-bottom: none;
    }
    .css-183lzff {
        color: white;
    }
    /* Custom card-like elements */
    .card {
        border-radius: 5px;
        background-color: #1a1f29;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        color: #1F77B4;
        font-size: 1.2em;
        margin-bottom: 10px;
        border-bottom: 1px solid #2c3e50;
        padding-bottom: 8px;
    }
    .section-container {
        background-color: #161b25;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1F77B4;
    }
    .success-message {
        background-color: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
        padding: 10px;
        border-radius: 4px;
    }
    .info-message {
        background-color: rgba(52, 152, 219, 0.2);
        border-left: 4px solid #3498db;
        padding: 10px;
        border-radius: 4px;
    }
    .warning-message {
        background-color: rgba(241, 196, 15, 0.2);
        border-left: 4px solid #f1c40f;
        padding: 10px;
        border-radius: 4px;
    }
    .sidebar-content {
        background-color: #1a1f29;
        border-radius: 5px;
        padding: 15px;
        margin-top: 20px;
    }
    h1, h2, h3, h4 {
        color: #1F77B4;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #1F77B4 0%, #3498db 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a custom title with icon
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Legal Assistant AI ⚖️</h1>
        <p>Your AI-powered legal expert for Indian law</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar with app information
with st.sidebar:
    st.markdown("## 🧑‍⚖️ Legal Assistant AI")
    st.title("About This App")
    st.markdown(
        """
        <div class="sidebar-content">
        <h3>Features</h3>
        <ul>
            <li>🤖 BNS Legal Query Assistant</li>
            <li>📄 Legal Document Validator</li>
            <li>⚡ Powered by Gemini AI & Pinecone</li>
            <li>📝 Generate Legal Drafts</li>
        </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="sidebar-content">
        <h3>How It Works</h3>
        <p>This app uses advanced AI and vector databases to provide accurate legal information and document validation based on Indian law.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="sidebar-content" style="margin-top: 40px;">
        <h4>Disclaimer</h4>
        <p style="font-size: 0.8em;">This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified legal professional for specific legal matters.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📚 Legal Query Assistant", "📄 Document Validator"])

with tab1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">BNS Legal Query Assistant</div>
            <p>Enter your legal query related to the Bharatiya Nyaya Sanhita (BNS) below and get detailed legal guidance with precise section references.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    query = st.text_area("Enter your legal query:", height=120, max_chars=500)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        query_button = st.button("🔍 Get Legal Answer", use_container_width=True)
    
    if query_button and query:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        with st.spinner("📚 Retrieving relevant legal provisions..."):
            results = retrieve_documents(query)
            context = process_results(results)
        
        with st.spinner("⚖️ Analyzing legal context and generating response..."):
            response = query_llm(query, context)
        
        st.subheader("Legal Analysis & Guidance")
        st.markdown(response)
        st.markdown('<div class="info-message">Note: The above response is based on the Bharatiya Nyaya Sanhita and related Indian laws.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Legal Document Validator</div>
            <p>Upload any legal document (PDF) to validate its contents, detect discrepancies, and generate a downloadable revised draft.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        file_path = Path(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Preview the uploaded file
        with st.expander("📄 Preview Uploaded Document", expanded=True):
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" style="border: none;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            validate_button = st.button("🚀 Validate Document", use_container_width=True)
        
        if validate_button:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            with st.spinner("🔍 Detecting document type and retrieving legal context..."):
                # First identify what type of document this is
                result, doc_type = process_document(file_path)
                doc_type_display = {
                    "divorce_petition": "Divorce Petition", 
                    "rental_agreement": "Rental Agreement", 
                    "general": "Legal Document"
                }
                
                # Display detected document type
                st.success(f"Document Type Detected: {doc_type_display.get(doc_type, 'Legal Document')}")
            
            # Clean up the output
            result = result.replace("**", "").strip()
            
            # Create tabs for different sections of the result
            result_tabs = st.tabs(["Summary", "Discrepancies", "Incorrect Clauses", "Corrected Clauses", "Missing Clauses", "Draft"])
            
            with result_tabs[0]:
                summary = extract_section(result, "Summary")
                st.markdown(f"<div class='success-message'>{summary}</div>", unsafe_allow_html=True)
            
            with result_tabs[1]:
                discrepancies = extract_section(result, "Discrepancies")
                st.markdown(discrepancies)
            
            with result_tabs[2]:
                incorrect_clauses = extract_section(result, "Incorrect Clauses")
                st.markdown(incorrect_clauses)
            
            with result_tabs[3]:
                corrected_clauses = extract_section(result, "Corrected Clauses")
                st.markdown(corrected_clauses)
            
            with result_tabs[4]:
                missing_clauses = extract_section(result, "Missing Clauses")
                st.markdown(missing_clauses)
            
            with result_tabs[5]:
                draft_text = extract_section(result, "Draft")
                st.text_area("Generated Draft", draft_text, height=400)
                
                # Create a Word document from the draft
                docx_filename = {
                    "divorce_petition": "mutual_consent_divorce_petition.docx",
                    "rental_agreement": "rental_agreement.docx",
                    "general": "legal_document.docx"
                }.get(doc_type, "legal_document.docx")
                
                doc_file = create_word_document(draft_text, doc_type, filename=docx_filename)
                
                # Provide download button for the Word document
                with open(doc_file, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Draft as Word Document",
                        data=file,
                        file_name=docx_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                
                st.markdown('<div class="info-message">The draft has been generated based on the document analysis and relevant legal provisions. You can download it using the button above.</div>', unsafe_allow_html=True)
            
            # Clean up the temporary file
            if file_path.exists():
                file_path.unlink()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #2c3e50;">
    <p style="color: #7f8c8d; font-size: 0.8em;">© 2025 Legal Assistant AI | For informational purposes only</p>
</div>
""", unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    pass  # Everything is handled by Streamlit's script running