import sys
sys.path.insert(0, r"e:\E桌面\AgentVNE")
sys.stdout.reconfigure(line_buffering=True)
print("start", flush=True)
try:
    from model import SimuVNE
    print("model imported OK", flush=True)
except Exception as e:
    print(f"model import failed: {e}", flush=True)
try:
    from torch_geometric.data import Data
    print("torch_geometric OK", flush=True)
except Exception as e:
    print(f"torch_geometric failed: {e}", flush=True)
