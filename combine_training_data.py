import json

# Load merged.json (qa_pairs format)
with open('merged.json', 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

# Load merged_training_data.json (training format)
with open('merged_training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

# Convert merged.json qa_pairs to training format
converted_data = []
for qa in merged_data['qa_pairs']:
    converted_entry = {
        "query": qa['question'],
        "response": qa['answer'],
        "source": f"Batch {merged_data['metadata']['batch']} - {qa['category']}",
        "category": qa['intent'] if 'intent' in qa else qa['category']
    }
    converted_data.append(converted_entry)

print(f"Converted {len(converted_data)} entries from merged.json")
print(f"Loaded {len(training_data)} entries from merged_training_data.json")

# Combine both datasets
combined = training_data + converted_data

# Remove duplicates based on query (case-insensitive comparison)
seen_queries = {}
unique_data = []

for entry in combined:
    query_lower = entry['query'].lower().strip()
    if query_lower not in seen_queries:
        seen_queries[query_lower] = True
        unique_data.append(entry)
    else:
        print(f"Duplicate found: '{entry['query'][:50]}...'")

print(f"\nTotal entries after removing duplicates: {len(unique_data)}")
print(f"Duplicates removed: {len(combined) - len(unique_data)}")

# Save the final combined training data
with open('final_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(unique_data, f, indent=2, ensure_ascii=False)

print(f"\nFinal training data saved to 'final_training_data.json' with {len(unique_data)} entries")
