import sys; sys.stdout.reconfigure(encoding='utf-8')

# Fix 1: dataset_generate_1.py
path1 = r"e:\E桌面\AgentVNE\dataset_generate_1.py"
with open(path1, "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace("'workflow_topo', 'workflow1_topo.json'", "'Workflow_topo', 'workflow1_topo.json'")
c = c.replace("'workflow_topo', 'workflow1_noderank.json'", "'Workflow_topo', 'workflow1_noderank.json'")
with open(path1, "w", encoding="utf-8") as f:
    f.write(c)
print("Fixed dataset_generate_1.py")

# Fix 2: model__sigmoid.py
path2 = r"e:\E桌面\AgentVNE\model__sigmoid.py"
with open(path2, "r", encoding="utf-8") as f:
    c = f.read()
old = "def create_model(input_dim=6, hidden_dim=64, hist_dim=32):"
new = "def create_model(input_dim=6, hidden_dim=64, hist_dim=32, num_nodes_j=10):"
c = c.replace(old, new)
with open(path2, "w", encoding="utf-8") as f:
    f.write(c)
print("Fixed model__sigmoid.py")
