# ✅ Extract the response JSON from GPT-4o completion
completion_result = completion.choices[0].message.parsed  # Extract parsed result

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

