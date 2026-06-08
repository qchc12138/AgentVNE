import re

path = r"e:\E桌面\AgentVNE\pretrain.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

original = content
changes = []

# 1. Add import math
old = "import os\nimport json"
new = "import os\nimport math\nimport json"
if old in content:
    content = content.replace(old, new, 1)
    changes.append("1. import math added")
else:
    changes.append("1. import math SKIPPED (not found)")

# 2. Change step_size=20 to step_size=50
old = "step_size=20"
new = "step_size=50"
if old in content:
    content = content.replace(old, new, 1)
    changes.append("2. step_size 20->50")
else:
    changes.append("2. step_size SKIPPED")

# 3. Add patience_counter after best_val_loss
old = "self.best_val_loss = float('inf')"
new = "self.best_val_loss = float('inf')\n        self.patience_counter = 0"
if old in content:
    content = content.replace(old, new, 1)
    changes.append("3. patience_counter added")
else:
    changes.append("3. patience_counter SKIPPED")

# 4. Change train() signature
old = "def train(self, num_epochs):"
new = "def train(self, num_epochs, patience=15):"
if old in content:
    content = content.replace(old, new, 1)
    changes.append("4. train signature updated")
else:
    changes.append("4. train signature SKIPPED")

# 5. Add best_epoch after start_time = time.time()
old = "start_time = time.time()\n\n        for epoch in range(num_epochs):"
new = "start_time = time.time()\n\n        best_epoch = 0\n\n        for epoch in range(num_epochs):"
if old in content:
    content = content.replace(old, new, 1)
    changes.append("5. best_epoch added")
else:
    changes.append("5. best_epoch SKIPPED")

# 6. Replace early-stop / save block
old = """            # 最优保存
            is_best = False
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  *** 新的最优验证损失: {val_loss:.6f} ***")
            self.save_checkpoint(epoch, val_loss if val_loss is not None else train_loss, is_best)"""

new_block = """            # 最优保存与早停
            is_best = False
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.patience_counter = 0
                    self.best_val_loss = val_loss
                    best_epoch = epoch + 1
                    print(f"  *** 新的最优验证损失: {val_loss:.6f} ***")
                else:
                    self.patience_counter += 1
                    print(f"  验证损失未改善 ({self.patience_counter}/{patience})")
                    if self.patience_counter >= patience:
                        print(f"\\n  早停触发! 最优验证损失: {self.best_val_loss:.6f} (epoch {best_epoch})")
                        self.save_checkpoint(epoch, val_loss, is_best)
                        break
            else:
                is_best = train_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_loss
                    best_epoch = epoch + 1
            self.save_checkpoint(epoch, val_loss if val_loss is not None else train_loss, is_best)"""

if old in content:
    content = content.replace(old, new_block, 1)
    changes.append("6. early-stop block replaced")
else:
    changes.append("6. early-stop SKIPPED (not found)")

# 7. Fix argparse defaults
content = content.replace(
    "parser.add_argument('--batch_size', type=int, default=100,",
    "parser.add_argument('--batch_size', type=int, default=16,", 1)
changes.append("7a. batch_size default 100->16")

content = content.replace(
    "parser.add_argument('--num_epochs', type=int, default=8,",
    "parser.add_argument('--num_epochs', type=int, default=100,", 1)
changes.append("7b. num_epochs default 8->100")

content = content.replace(
    "parser.add_argument('--learning_rate', type=float, default=0.005,",
    "parser.add_argument('--learning_rate', type=float, default=0.001,", 1)
changes.append("7c. learning_rate default 0.005->0.001")

# 8. Replace create_pretrain_dataloader docstring and add new function after it
old = '''def create_pretrain_dataloader(samples: List[Dict], batch_size: int = 16):
    """创建仅训练用的 DataLoader（不划分验证集）。"""
    from torch.utils.data import DataLoader
    train_loader = DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=_collate_samples)
    return train_loader'''

new_func = '''def create_pretrain_dataloader(samples: List[Dict], batch_size: int = 16):
    """创建单个 DataLoader（不分 train/val）。保留兼容旧调用。"""
    from torch.utils.data import DataLoader
    train_loader = DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=_collate_samples)
    return train_loader


def create_train_val_dataloaders(samples: List[Dict], batch_size: int = 16, train_ratio: float = 0.8):
    """按比例拆分训练/验证集，返回两个 DataLoader。"""
    from torch.utils.data import DataLoader
    n = len(samples)
    n_train = int(np.ceil(n * train_ratio))
    indices = list(range(n))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    print(f"  训练集: {len(train_samples)} 样本, 验证集: {len(val_samples)} 样本")
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=_collate_samples)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False, collate_fn=_collate_samples)
    return train_loader, val_loader'''

if old in content:
    content = content.replace(old, new_func, 1)
    changes.append("8. create_train_val_dataloaders added")
else:
    changes.append("8. create func SKIPPED")

# 9. Fix main() dataloader creation
old = '''    # 创建数据加载器（全量作为训练集）
    print("\\n" + "="*60)
    print("创建训练数据加载器...")
    print("="*60)
    train_loader = create_pretrain_dataloader(samples=samples, batch_size=config['batch_size'])'''

new_main = '''    # 创建训练/验证数据加载器
    print("\\n" + "="*60)
    print("创建训练/验证数据加载器...")
    print("="*60)
    train_loader, val_loader = create_train_val_dataloaders(
        samples=samples, batch_size=config['batch_size'], train_ratio=0.8)'''

if old in content:
    content = content.replace(old, new_main, 1)
    changes.append("9. main dataloader updated")
else:
    changes.append("9. main dataloader SKIPPED")

# 10. Fix PretrainTrainer val_loader=None -> val_loader=val_loader
# The trainer is instantiated after model creation. Find the right occurrence.
old = 'val_loader=None,'
# In the main's PretrainTrainer call (near the end), replace val_loader=None with val_loader=val_loader
# But the function signature also has val_loader=None - we need to keep that.
# Strategy: replace the one that comes AFTER "model=model,"
idx = content.rfind("val_loader=None,")  # last occurrence is in main()
if idx > 0 and "model=model" in content[idx-200:idx]:
    content = content[:idx] + "val_loader=val_loader," + content[idx+len(old):]
    changes.append("10. val_loader=val_loader in main")
else:
    changes.append("10. val_loader SKIPPED")

for c in changes:
    print(c)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("\nFile written successfully")
