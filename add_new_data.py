import json

# Load the existing final training data
with open('final_training_data.json', 'r', encoding='utf-8') as f:
    final_data = json.load(f)

print(f"Loaded {len(final_data)} entries from final_training_data.json")

# Load new_data.json
# The file contains two separate JSON arrays, so we need to parse them separately
with open('new_data.json', 'r', encoding='utf-8') as f:
    content = f.read().strip()

    # Split by ][ to get the two arrays
    parts = content.split('][')

    new_data = []

    # Parse the first array (add the missing closing bracket)
    if len(parts) > 0:
        first_array = parts[0]
        if not first_array.startswith('['):
            first_array = '[' + first_array
        if not first_array.endswith(']'):
            first_array = first_array + ']'
        new_data.extend(json.loads(first_array))

    # Parse the second array (add the missing opening bracket)
    if len(parts) > 1:
        second_array = parts[1]
        if not second_array.startswith('['):
            second_array = '[' + second_array
        if not second_array.endswith(']'):
            second_array = second_array + ']'

        # Remove trailing commas before the closing bracket
        second_array = second_array.replace(',]', ']')
        second_array = second_array.replace(', ]', ']')

        new_data.extend(json.loads(second_array))

print(f"Loaded {len(new_data)} entries from new_data.json")

# Convert new_data to training format
converted_new_data = []
for entry in new_data:
    converted_entry = {
        "query": entry['question'],
        "response": entry['answer'],
        "source": entry.get('data', 'New Data Batch'),
        "category": entry.get('intent', entry.get('category', 'general'))
    }
    converted_new_data.append(converted_entry)

print(f"Converted {len(converted_new_data)} entries from new_data.json")

# Combine both datasets
combined = final_data + converted_new_data

# Remove duplicates based on query (case-insensitive comparison)
seen_queries = {}
unique_data = []
duplicates_found = 0

for entry in combined:
    query_lower = entry['query'].lower().strip()
    if query_lower not in seen_queries:
        seen_queries[query_lower] = True
        unique_data.append(entry)
    else:
        duplicates_found += 1
        print(f"Duplicate found: '{entry['query'][:60]}...'")

print(f"\nTotal entries after removing duplicates: {len(unique_data)}")
print(f"Duplicates removed: {duplicates_found}")

# Save the updated final training data
with open('final_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(unique_data, f, indent=2, ensure_ascii=False)

print(f"\nUpdated training data saved to 'final_training_data.json' with {len(unique_data)} entries")
print(f"Added {len(unique_data) - len(final_data)} new entries")
