# Cell - 0
import os
import PyPDF2
import json
import urllib.request
import base64
import requests
import time
import pandas as pd
import io

from openai import AzureOpenAI
from pydantic import create_model
from llm_confidence.logprobs_handler import LogprobsHandler
from IPython.display import display
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature, AnalyzeDocumentRequest, DocumentContentFormat
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import dotenv_values
from pdf2image import convert_from_bytes
from collections import defaultdict



from modules.app_settings import AppSettings
from modules.utils import Stopwatch
from modules.accuracy_evaluator import AccuracyEvaluator
from modules.comparison import get_extraction_comparison
from modules.confidence import merge_confidence_values
from modules.document_intelligence_confidence import evaluate_confidence as di_evaluate_confidence
from modules.openai_confidence import evaluate_confidence as oai_evaluate_confidence
from modules.document_processing_result import DataExtractionResult

accuracy_evaluator = AccuracyEvaluator(match_keys=[])


#Cell - 1
# ✅ Initialize Azure Document Intelligence Client
endpoint = "YOUR_AZURE_ENDPOINT"
key = "YOUR_AZURE_KEY"

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

# ✅ Function to Detect File Type (PDF or Image)
def get_content_type(uploaded_file):
    extension = uploaded_file.name.lower().split('.')[-1]
    if extension == "pdf":
        return "application/pdf"
    elif extension in ["jpg", "jpeg", "png"]:
        return "image/jpeg"
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or Image.")

# ✅ Function to Process the Uploaded Invoice (PDF or Image)
def prebuilt_invoice(uploaded_file):
    """
    Processes an uploaded invoice (PDF or Image) using Azure Document Intelligence.
    Extracts structured key-value pairs, tables, and layout details.

    :param uploaded_file: The uploaded invoice file from Streamlit.
    :return: Extracted structured invoice data.
    """

    extracted_data = []  # ✅ List to store extracted invoice details

    # ✅ Detect Content Type (PDF or Image)
    content_type = get_content_type(uploaded_file)

    # ✅ Process the Uploaded File Directly
    with uploaded_file:  # ✅ Read File In-Memory (No Saving to Disk)
        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-invoice",
            body=uploaded_file,  # ✅ Use Uploaded File Directly
            content_type=content_type
        )

    result = poller.result()  # ✅ Get analysis result

    invoice_data = {}  # ✅ Dictionary to store extracted key-value pairs
    tables_data = []   # ✅ List to store extracted table data

    # ✅ Extract Key-Value Pairs
    if result.documents:
        for invoice in result.documents:
            for field_name, field_value in invoice.fields.items():
                if field_value:
                    invoice_data[field_name] = {
                        "value": field_value.content,
                    }

    # ✅ Extract Tables
    for table in result.tables:
        table_rows = defaultdict(dict)
        headers = []

        # ✅ Identify Table Headers
        first_row_index = min(cell.row_index for cell in table.cells)
        for cell in table.cells:
            if cell.row_index == first_row_index:
                headers.append(cell.content)

        # ✅ Extract Table Data
        for cell in table.cells:
            row_index = cell.row_index
            col_index = cell.column_index

            column_name = headers[col_index] if col_index < len(headers) else f"Column_{col_index + 1}"
            table_rows[row_index][column_name] = cell.content

        formatted_rows = [table_rows[row_idx] for row_idx in sorted(table_rows.keys())]

        tables_data.append({
            "table_number": len(tables_data) + 1,
            "headers": headers,
            "rows": formatted_rows
        })

    # ✅ Extract Key-Value Pairs Found in Document
    key_value_pairs = {}
    if result.key_value_pairs:
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key_value_pairs[kv_pair.key.content] = {
                    "value": kv_pair.value.content,
                }

    # ✅ Store extracted data in memory (NOT saving to a file)
    extracted_data.append({
        "invoice_fields": invoice_data,
        "key_value_pairs": key_value_pairs,
        "tables": tables_data
    })

    return extracted_data  # ✅ Now directly returning data


#Cell 2
def generate_schema_from_json(extracted_data: list, schema_name="InvoiceSchema"):
    """
    Dynamically generates a Pydantic schema from extracted invoice data.

    :param extracted_data: The extracted invoice data from `prebuilt_invoice()`.
    :param schema_name: The name of the schema.
    :return: A dynamically generated Pydantic model.
    """
    if not extracted_data:
        raise ValueError("Extracted data is empty. Cannot generate schema.")

    fields = {}
    first_invoice = extracted_data[0]  # ✅ Use in-memory extracted data

    # ✅ Extract invoice fields
    for key, value in first_invoice.get("invoice_fields", {}).items():
        if isinstance(value.get("value"), str):
            fields[key] = (str, None)
        elif isinstance(value.get("value"), (float, int)):
            fields[key] = (float, None)
        else:
            fields[key] = (str, None)  # Default unknown types to string

    # ✅ Extract key-value pairs
    for key, value in first_invoice.get("key_value_pairs", {}).items():
        fields[key] = (str, None)

    # ✅ Create a dynamic Pydantic model
    DynamicSchema = create_model(schema_name, **fields)
    return DynamicSchema


#Cell 3
def process_invoice(uploaded_pdf):
    """
    Extracts structured data from an uploaded invoice PDF and prepares it for accuracy evaluation.

    :param uploaded_pdf: The uploaded PDF file from Streamlit.
    :return: Extracted structured data (expected values).
    """

    # ✅ Step 1: Extract invoice data using `prebuilt_invoice()` (Process PDF in memory)
    extracted_data = prebuilt_invoice(uploaded_pdf)  # No JSON file saving

    # ✅ Step 2: Generate schema dynamically from extracted invoice data
    invoice_schema = generate_schema_from_json(extracted_data)  # Uses extracted data

    # ✅ Step 3: Extract and clean invoice key-value fields
    cleaned_invoice_data = {
        key: (value["value"] if isinstance(value, dict) and "value" in value else value) or ""
        for key, value in extracted_data[0]["invoice_fields"].items()
    }

    # ✅ Step 4: Extract and clean table data (keep only values)
    cleaned_table_data = []
    if "tables" in extracted_data[0]:
        for table in extracted_data[0]["tables"]:
            cleaned_rows = []
            for row in table["rows"]:
                cleaned_row = {col: (val["value"] if isinstance(val, dict) and "value" in val else val) or "" for col, val in row.items()}
                cleaned_rows.append(cleaned_row)
            cleaned_table_data.append({"table_number": table["table_number"], "headers": table["headers"], "rows": cleaned_rows})

    # ✅ Step 5: Merge extracted fields & tables into `expected`
    expected_data = {
        **cleaned_invoice_data,  # ✅ Invoice Key-Value Pairs
        "tables": cleaned_table_data  # ✅ Invoice Tables
    }

    # ✅ Step 6: Pass cleaned data to schema (ensures consistent format)
    expected = invoice_schema(**expected_data)

    return expected  # ✅ Returns structured extracted invoice data

# ✅ Cell 4: Prebuilt-Layout Model Processing (for OCR + Document Layout Extraction)
def process_layout_model(uploaded_pdf):
    """
    Uses the Azure Prebuilt-Layout model to extract structured text, tables, and layout details.

    :param uploaded_pdf: The uploaded file (PDF or image) from Streamlit.
    :return: Extracted markdown content from the document.
    """

    # ✅ Determine the correct content type (PDF or Image)
    content_type = get_content_type(uploaded_pdf.name)

    with Stopwatch() as di_stopwatch:
        with uploaded_pdf as f:
            poller = document_intelligence_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                output_content_format=DocumentContentFormat.MARKDOWN,
                content_type=content_type
            )

        result: AnalyzeResult = poller.result()

    # ✅ Extract markdown content (includes structured text, tables, etc.)
    markdown = result.content

    return markdown, result  # ✅ Pass extracted content for downstream processing


#Cell 5 -
# ✅ Step 1: Set Up Azure OpenAI Credentials
api_key = "YOUR_AZURE_OPENAI_KEY"
api_version = "YOUR_AZURE_OPENAI_API_VERSION"
azure_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"

# ✅ Step 2: Initialize OpenAI Client
openai_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

with Stopwatch() as oai_stopwatch:
    completion = openai_client.beta.chat.completions.parse(
        model=settings.gpt4o_model_deployment_name,
        messages=[
            {
                "role": "system",
                "content": """Extract structured data from this document.
    - If a value is missing, return **null**.
    - Maintain all detected fields as they appear in the document.
    - Preserve the structure of tables, ensuring correct alignment of columns and values.
    - Do not infer missing values beyond what is available in the document.
    - Return the extracted information in **JSON format** where:
      - Keys should match the field names detected in the document.
      - Values should be extracted as they appear in the document.
      - If a field is a table, **maintain column consistency** and return it as a structured list of dictionaries.

### **Additional Instructions**
- Preserve numerical values **as they appear in the document**. Do not apply rounding, unit conversion, or add thousand separators.
- Keep company and product names exactly as they appear in the document.
- Ensure correct data mapping between labels and values.

### **Output Format**
Return the extracted user-specified fields as a **JSON object**, keeping all fields dynamically based on the document content.
"""
            },
            {
                "role": "user",
                "content": f"Use this content to provide the response{markdown}"},
        ],
        response_format=invoice_schema,  # Uses the dynamically generated schema instead of a fixed one
        max_tokens=4096,
        temperature=0.1,
        top_p=0.1,
        logprobs=True
    )

raw_response = completion.choices[0].message.content  # ✅ Extract the raw response

# Cell - 6
# ✅ Extract structured invoice data from GPT-4o completion response
invoice_data = completion.choices[0].message.parsed

# ✅ Convert expected invoice schema to a dictionary (from JSON schema)
expected_dict = expected.model_dump()  # ✅ Use `.model_dump()` instead of `.to_dict()`

# ✅ Convert extracted invoice data to a dictionary
invoice_data_dict = invoice_data.model_dump()  # ✅ Use `.model_dump()` instead of `.to_dict()`

# Cell - 7
# ✅ Evaluate accuracy using global accuracy_evaluator
accuracy = accuracy_evaluator.evaluate(expected=expected_dict, actual=invoice_data_dict)

#cell 8
# Evaluate confidence from Azure Document Intelligence (Prebuilt-Layout Model)
di_confidence = di_evaluate_confidence(invoice_data_dict, result)

# Evaluate confidence from GPT-4o Vision extraction
oai_confidence = oai_evaluate_confidence(invoice_data_dict, completion.choices[0])

# Merge both confidence scores into a final confidence metric
confidence = merge_confidence_values(di_confidence, oai_confidence)

# Calculate total execution time (Prebuilt-Layout + Image Processing + GPT-4o)
total_elapsed = di_stopwatch.elapsed + oai_stopwatch.elapsed

# Retrieve token usage from GPT-4o completion response
prompt_tokens = completion.usage.prompt_tokens
completion_tokens = completion.usage.completion_tokens

#Cell 9 
def process_final_extraction(invoice_data_dict, confidence, accuracy, prompt_tokens, completion_tokens, total_elapsed):
    """
    Processes the final extracted invoice data without saving to a JSON file.
    
    :param invoice_data_dict: Extracted structured invoice data.
    :param confidence: Confidence scores from both AI models.
    :param accuracy: Accuracy evaluation results.
    :param prompt_tokens: Number of tokens used in prompt.
    :param completion_tokens: Number of tokens used in completion.
    :param total_elapsed: Total processing time.
    :return: A structured DataExtractionResult object.
    """
    
    # ✅ Create the final extraction result object
    extraction_result = DataExtractionResult(
        invoice_data_dict, confidence, accuracy, prompt_tokens, completion_tokens, total_elapsed
    )

    # ✅ Return structured extraction result for further use
    return extraction_result  # ✅ No JSON saving, just returning the structured object

# Cell 10 - 
def generate_extraction_summary(accuracy, confidence, total_elapsed, di_stopwatch, oai_stopwatch, prompt_tokens, completion_tokens, expected_dict, invoice_data_dict):
    """
    Generates a summary DataFrame of the extracted invoice data accuracy, confidence, and execution time.

    :param accuracy: Dictionary containing overall accuracy metrics.
    :param confidence: Dictionary containing confidence scores.
    :param total_elapsed: Total processing time.
    :param di_stopwatch: Stopwatch instance for Document Intelligence processing time.
    :param oai_stopwatch: Stopwatch instance for OpenAI processing time.
    :param prompt_tokens: Number of tokens used in the prompt.
    :param completion_tokens: Number of tokens used in completion.
    :param expected_dict: Expected structured invoice data.
    :param invoice_data_dict: Extracted structured invoice data.
    :return: A tuple containing (summary DataFrame, extraction comparison DataFrame).
    """

    # ✅ Create a summary DataFrame
    df = pd.DataFrame([{
        "Accuracy": f"{accuracy['overall'] * 100:.2f}%",
        "Confidence": f"{confidence['_overall'] * 100:.2f}%",
        "Execution Time": f"{total_elapsed:.2f} seconds",
        "Document Intelligence Execution Time": f"{di_stopwatch.elapsed:.2f} seconds",
        "OpenAI Execution Time": f"{oai_stopwatch.elapsed:.2f} seconds",
        "Prompt Tokens": prompt_tokens,
        "Completion Tokens": completion_tokens
    }])

    # ✅ Generate extraction comparison table (without styling)
    styled_comparison = get_extraction_comparison(expected_dict, invoice_data_dict, confidence, accuracy['accuracy'])

    # ✅ Convert to normal DataFrame if `get_extraction_comparison()` returns a styled object
    if isinstance(styled_comparison, pd.io.formats.style.Styler):
        styled_comparison = styled_comparison.data  

    return df, styled_comparison  # ✅ Return both summary and comparison table

#Cell 11 - 
# ✅ Ensure `user_query` comes from the Streamlit app dynamically
def extract_invoice_data(user_query):
    """
    Uses GPT-4o to extract structured invoice data based on a user query.
    
    :param user_query: The query entered by the user in Streamlit (e.g., "Extract only invoice date and Invoice Total").
    :return: Extracted structured response from GPT-4o.
    """

    completion = openai_client.beta.chat.completions.parse(
        model=settings.gpt4o_model_deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"""Extract structured data from the document with these column names - **Field**, **Expected**, **Extracted**, **Confidence**, **Accuracy** present in {styled_comparison}
                
                "Use this content to provide the tabular response for the Query: {styled_comparison}"
    - If a value is missing, return **null**.
    - Maintain all detected fields as they appear in the document.
    - Preserve the structure of tables, ensuring correct alignment of columns and values.
    - Do not infer missing values beyond what is available in the document.

    ### **Output Format**
    Return the extracted invoice data as a **JSON object**, keeping all fields dynamically based on the document content.
    """
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Query: {user_query}"}]  # ✅ Dynamically pass the user's query
            }
        ],
        max_tokens=4096,
        temperature=0.1,
        top_p=0.1,
        logprobs=True
    )

    raw_response = completion.choices[0].message.content  # ✅ Extract response
    return raw_response  # ✅ Return extracted data

