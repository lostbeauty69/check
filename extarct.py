import json
import pandas as pd
from IPython.display import display, Markdown

# ✅ Extract raw response content
raw_response = completion.choices[0].message.content  # Extract raw response

# ✅ Debugging: Display GPT-4o Response
print("\n🔹 **Raw GPT-4o Response (Markdown Format)**")
display(Markdown(str(raw_response)))  # Show response as Markdown

# ✅ Convert raw response string to JSON (if formatted correctly)
try:
    completion_result = json.loads(raw_response)  # Convert response string to JSON
except json.JSONDecodeError:
    print("❌ GPT-4o did not return a valid JSON format.")
    completion_result = None

# ✅ Convert JSON response to DataFrame format
if isinstance(completion_result, dict):
    extracted_data_df = pd.DataFrame.from_dict(completion_result, orient="index")
    extracted_data_df.columns = ["Expected", "Extracted", "Confidence", "Accuracy"]
    extracted_data_df.insert(0, "Field", extracted_data_df.index)  # Add 'Field' column
    extracted_data_df.reset_index(drop=True, inplace=True)

    # ✅ Display the extracted table in a structured format
    print("\n🔹 **Extracted Query Data from GPT-4o**")
    display(extracted_data_df)
else:
    print("❌ No structured data extracted from the GPT-4o response.")
