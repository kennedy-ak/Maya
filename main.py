import json

# Load the three JSON files
with open('batch5.json', 'r') as f:
    data1 = json.load(f)

with open('batch4.json', 'r') as f:
    data2 = json.load(f)

# with open('batch3.json', 'r') as f:
#     data3 = json.load(f)

# Combine the qa_pairs arrays
combined_qa_pairs = data1['qa_pairs'] + data2['qa_pairs']

# Create the merged structure with metadata from the first file
merged_data = {
    "metadata": data1['metadata'],
    "qa_pairs": combined_qa_pairs
}

# Update metadata to reflect the merge
merged_data['metadata']['total_in_batch'] = len(combined_qa_pairs)
merged_data['metadata']['batch'] = "merged"

# Write the combined data to a new JSON file
with open('merged.json', 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged JSON files into 'merged_training_data.json' with {len(combined_qa_pairs)} QA pairs")
