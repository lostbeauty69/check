# Cell 2 - Fully dynamic extraction of individual items
def generate_schema_from_json(json_path: str, schema_name="InvoiceSchema"):
    try:
        with open(json_path, "r") as f:
            extracted_data = json.load(f)  # Load extracted JSON
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")

    fields = {}
    first_invoice = extracted_data[0]  # Get first invoice

    # ✅ Process invoice fields normally (excluding "Items" table for now)
    for key, value in first_invoice.get("invoice_fields", {}).items():
        if key == "Items" and isinstance(value, list):  # Detect table dynamically
            # ✅ Extract column names from the first row of the items table
            if value:  # Ensure there are items
                column_names = list(value[0].keys())  # Extract headers dynamically

                # ✅ Create dynamic fields based on the number of items
                for idx, item in enumerate(value):  # Loop through rows (items)
                    for col in column_names:  # Loop through dynamic columns
                        fields[f"Item_{idx+1}_{col}"] = (str, None)  # Dynamically add fields
        elif isinstance(value.get("value"), str):
            fields[key] = (str, None)
        elif isinstance(value.get("value"), (float, int)):
            fields[key] = (float, None)
        else:
            fields[key] = (str, None)

    # ✅ Process key-value pairs (No change)
    for key, value in first_invoice.get("key_value_pairs", {}).items():
        fields[key] = (str, None)

    return create_model(schema_name, **fields)

invoice_schema = generate_schema_from_json("extracted_invoices/mrinmoy.json")
