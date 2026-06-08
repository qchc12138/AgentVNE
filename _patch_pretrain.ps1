$path = "e:\E桌面\AgentVNE\pretrain.py"
$content = Get-Content $path -Raw -Encoding UTF8
$lines = $content -split "\r?\n"

# Show lines around the train method early-stop block
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "if val_loss is not None:") {
        Write-Host "--- Found at line $($i+1) ---"
        for ($j = [Math]::Max(0, $i - 2); $j -lt [Math]::Min($lines.Count, $i + 20); $j++) {
            Write-Host "$($j+1): $($lines[$j])"
        }
        break
    }
}
