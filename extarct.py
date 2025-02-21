import streamlit as st
import pandas as pd
from Hybrid_IDP_VLM_model import extract_invoice_data  # ✅ Import function that returns final table result

# Streamlit UI Configuration
st.set_page_config(page_title="Invoice Extraction App", page_icon="📄")
st.title("📄 Invoice Extraction Using AI Models")
st.write("Upload a document and enter a query to extract structured data.")

# File Uploader
uploaded_file = st.file_uploader("Upload your Invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
user_query = st.text_input("Enter your query (e.g., Extract Invoice Date & Total)")

# Process the user query and display results
if uploaded_file and user_query:
    with st.spinner("Processing your request..."):
        response = extract_invoice_data(user_query)  # ✅ Get the final table result

    # ✅ Display the extracted table
    if response:
        st.write("### 📌 Extracted Table (User Query Result)")

        # ✅ Check if response is JSON and convert it to a DataFrame
        try:
            df_response = pd.read_json(response)  # ✅ Convert JSON response to DataFrame
            st.table(df_response)  # ✅ Display the table in Streamlit
        except:
            st.write(response)  # ✅ If JSON conversion fails, display raw response

