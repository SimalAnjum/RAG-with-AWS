import json
import os
from collections import defaultdict

# Paths
input_path = "datasets/cuad_raw/CUADv1.json"
output_dir = "datasets/processed_json"
os.makedirs(output_dir, exist_ok=True)

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Process each document
grouped = defaultdict(lambda: {"qa_pairs": [], "title": None})

for item in full_data["data"]:
    doc_id = item["title"]
    grouped[doc_id]["title"] = doc_id
    grouped[doc_id]["document_id"] = doc_id
    grouped[doc_id]["source"] = "CUAD"

    for paragraph in item["paragraphs"]:
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"] if qa["answers"] else ""
            grouped[doc_id]["qa_pairs"].append({
                "question": question,
                "answer": answer
            })

# Save each doc as its own JSON
for doc_id, content in grouped.items():
    with open(os.path.join(output_dir, f"{doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

print(f"✅ Processed {len(grouped)} documents into {output_dir}")
