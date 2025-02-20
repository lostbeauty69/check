# ✅ User Query (Dynamic) - Replace with user input
user_query = "Extract only Invoice Date and Invoice Total from the content provided"

# ✅ Extract matching fields dynamically from `styled_comparison`
query_keywords = [word.strip() for word in user_query.replace("Extract only", "").split(" and ")]
filtered_comparison = styled_comparison[styled_comparison["Field"].str.contains('|'.join(query_keywords), case=False, na=False)]

# ✅ Ensure at least some results are found, else return "No results found"
if filtered_comparison.empty:
    display(pd.DataFrame({"Message": ["No matching fields found for the query."]}))
else:
    # ✅ Prepare extracted data for GPT-4o
    extracted_text = filtered_comparison.to_string(index=False)

    # ✅ Run GPT-4o to format the response dynamically
    with Stopwatch() as oai_stopwatch:
        completion = openai_client.beta.chat.completions.parse(
            model=settings.gpt4o_model_deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": f"""Extract structured data from the document with these column names:
                    - **Field**, **Expected**, **Extracted**, **Confidence**, **Accuracy**.

                    - Only return relevant fields based on the user query.
                    - Ensure data is structured and aligned in a tabular format.
                    - Do not infer missing values beyond what is available.

                    ### **Output Format**
                    Return the extracted invoice data as a structured table, formatted in JSON.
                    """
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Provide structured data for this query:\n{user_query}\n\n{extracted_text}"}]
                }],
            max_tokens=4096,
            temperature=0.1,
            top_p=0.1,
            logprobs=True
        )

    # ✅ Convert GPT-4o response into a DataFrame
    extracted_response = completion.choices[0].message.parsed  # Extract structured JSON response

    # ✅ Convert response into a Pandas DataFrame
    extracted_table = pd.DataFrame.from_dict(extracted_response, orient="index")

    # ✅ Rename columns for clarity (Ensure it matches Cell 13 format)
    extracted_table.columns = ["Expected", "Extracted", "Confidence", "Accuracy"]

    # ✅ Add Field column to keep structure consistent
    extracted_table.insert(0, "Field", extracted_table.index)

    # ✅ Reset index for clean formatting
    extracted_table.reset_index(drop=True, inplace=True)

    # ✅ Display extracted response in a **proper table format** matching Cell 13
    display(extracted_table)
