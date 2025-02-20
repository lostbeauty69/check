
# 🎨 **Streamlit UI Design**
st.set_page_config(page_title="Invoice Extraction App", page_icon="📄")

st.title("📄 Invoice Extraction Using AI Models")
st.write("Upload a document and select a model to extract structured data.")

# 📤 **File Uploader**
uploaded_file = st.file_uploader("Upload your Invoice (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

# ✏️ **User Query Input**
user_query = st.text_input("Enter your query (e.g., Extract Invoice Date & Vendor Address)")

# 🚀 **Model Selection Buttons**
st.write("### Choose an AI Model for Extraction")

col1, col2, col3, col4 = st.columns(4)

# ✅ Process the uploaded file (convert to bytes if needed)
if uploaded_file:
    file_bytes = uploaded_file.read()  # Read file content as bytes

if uploaded_file and user_query:
    response = None  # Placeholder for response

    # ✅ **VLM using Gemini**
    with col1:
        if st.button("VLM using Gemini"):
            with st.spinner("Processing with Gemini..."):
                response = gemini_process(file_bytes, user_query)

    # ✅ **VLM using GPT-4o**
    with col2:
        if st.button("VLM using GPT-4o"):
            with st.spinner("Processing with GPT-4o..."):
                response = gpt4o_process(file_bytes, user_query)

    # ✅ **Hybrid IDP VLM Model**
    with col3:
        if st.button("Hybrid IDP VLM Model"):
            with st.spinner("Processing with Hybrid IDP VLM Model..."):
                response = hybrid_process(file_bytes, user_query)

    # ✅ **Hybrid IDP VLM Model for Images**
    with col4:
        if st.button("Hybrid IDP VLM Model for Images"):
            with st.spinner("Processing with Hybrid IDP VLM Model for Images..."):
                response = hybrid_images_process(file_bytes, user_query)

    # ✅ Display the Extracted Data
    if response:
        st.write("### 📌 Extracted Data")

        # Ensure response is in DataFrame format for proper display
        if isinstance(response, pd.DataFrame):
            st.table(response)
        elif isinstance(response, dict):  # Convert dict to DataFrame
            df_response = pd.DataFrame([response])
            st.table(df_response)
        elif isinstance(response, list) and all(isinstance(i, dict) for i in response):  # List of dicts
            df_response = pd.DataFrame(response)
            st.table(df_response)
        else:
            st.write(response)  # Fallback (for plain text)

# 📝 **Instructions**
st.write("### How to Use")
st.write("1. **Upload a document** (PDF or Image).")
st.write("2. **Enter a query** about the invoice details you need.")
st.write("3. **Click one of the model buttons** to extract the data.")
st.write("4. **View structured results in table format.**")
