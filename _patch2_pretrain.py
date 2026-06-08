path = r"e:\E桌面\AgentVNE\pretrain.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

changes = []

# Fix 5: add best_epoch after start_time (currently missing)
old5 = "start_time = time.time()\n\n        for epoch in range(num_epochs):"
new5 = "start_time = time.time()\n\n        best_epoch = 0\n\n        for epoch in range(num_epochs):"
if old5 in content:
    content = content.replace(old5, new5, 1)
    changes.append("5. best_epoch added")
else:
    # Try with different whitespace
    import re
    m = re.search(r'start_time = time\.time\(\)\s*\n\s*for epoch in range\(num_epochs\):', content)
    if m:
        print("Found at:", m.start(), "-", m.end())
        print("Match:", repr(m.group()))
        # Replace
        old_match = m.group()
        new_match = "start_time = time.time()\n\n        best_epoch = 0\n\n        for epoch in range(num_epochs):"
        content = content[:m.start()] + new_match + content[m.end():]
        changes.append("5. best_epoch added (regex)")
    else:
        changes.append("5. best_epoch FAILED")

# Fix 6: replace save/early-stop block with actual current content
old6 = """            # 保存检查点
            is_best = False
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  *** 新的最优验证损失: {val_loss:.6f} ***")
            self.save_checkpoint(epoch, val_loss if val_loss is not None else train_loss, is_best)"""

new6 = """            # 最优保存与早停
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

if old6 in content:
    content = content.replace(old6, new6, 1)
    changes.append("6. early-stop block replaced")
else:
    # Try regex
    import re
    pattern = r'(# 保存检查点\s*\n\s*is_best = False.*?self\.save_checkpoint\(epoch, val_loss if val_loss is not None else train_loss, is_best\))'
    m = re.search(pattern, content, re.DOTALL)
    if m:
        content = content[:m.start()] + new6 + content[m.end():]
        changes.append("6. early-stop block replaced (regex)")
    else:
        changes.append("6. early-stop FAILED")

for c in changes:
    print(c)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("\nDone")
