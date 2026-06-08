path = r"e:\E桌面\AgentVNE\pretrain.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix import: model_1 -> model
if "from model_1 import" in content:
    content = content.replace("from model_1 import", "from model import")
    print("Fixed model_1 -> model")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
