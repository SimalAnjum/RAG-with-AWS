# from datasets import load_dataset
# import json
# import os

# # Load the CUAD dataset from HuggingFace
# dataset = load_dataset("theatticusproject/cuad")

# # Convert HuggingFace Dataset object to a plain list
# data_list = dataset["train"].to_list()

# # Create output directory
# output_dir = "datasets/cuad_raw"
# os.makedirs(output_dir, exist_ok=True)

# # Save as JSON
# with open(os.path.join(output_dir, "cuad.json"), "w", encoding="utf-8") as f:
#     json.dump(data_list, f, indent=2)

# print("✅ CUAD dataset downloaded and saved to cuad.json")
