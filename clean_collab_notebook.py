import json

notebook_path = "collab.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

original_count = len(nb.get("cells", []))
new_cells = []

for cell in nb.get("cells", []):
    source_text = "".join(cell.get("source", []))
    if "SUPERVISED EVALUATION" in source_text or "Supervised_Evaluation_F1_Kappa" in source_text:
        print("Found supervised evaluation cell. Removing...")
        continue
    new_cells.append(cell)

nb["cells"] = new_cells
print(f"Removed cell(s). Cell count went from {original_count} to {len(new_cells)}.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("collab.ipynb successfully updated!")
