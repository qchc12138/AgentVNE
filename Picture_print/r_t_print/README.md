# r_t 随时间变化图绘制脚本使用说明

## 功能概述

`r_t_t.py` 是一个用于绘制 r_t 随时间变化的折线图的 Python 脚本。

**主要功能**：
- ✅ 从 JSON 数据文件读取 r_t 数据
- ✅ **控制时间轴的起始和结束位置**
- ✅ 选择显示哪些策略
- ✅ 选择是否显示绝对值
- ✅ 自定义图表样式（大小、分辨率、线宽等）

---

## 快速开始

### 基本用法

```bash
# 绘制所有策略，时间步 0-1000
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 0 --end 1000

# 绘制指定策略，时间步 100-500
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 100 --end 500 --strategies ga gal

# 显示原始 r_t 值（不使用绝对值）
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 0 --end 1000 --no-abs

# 指定输出文件
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 0 --end 1000 --output my_plot.png
```

---

## 参数说明

### 必需参数

- `--input` / `-i`: 输入 JSON 数据文件路径（必需）

### 可选参数

#### 时间轴控制

- `--start` / `-s`: 起始时间步（包含），默认：0
- `--end` / `-e`: 结束时间步（包含），默认：使用数据文件中的 `max_time_steps`

**示例**：
```bash
# 只显示前 100 个时间步
python3 r_t_t.py --input data.json --start 0 --end 100

# 显示中间部分（100-500）
python3 r_t_t.py --input data.json --start 100 --end 500

# 只指定起始时间，结束时间自动使用数据文件中的 max_time_steps
python3 r_t_t.py --input data.json --start 500
```

#### 策略选择

- `--strategies` / `-st`: 要显示的策略列表，默认：显示所有策略

**示例**：
```bash
# 只显示 ga 和 gal 策略
python3 r_t_t.py --input data.json --strategies ga gal

# 只显示 pretrain 策略
python3 r_t_t.py --input data.json --strategies pretrain
```

#### 显示选项

- `--no-abs`: 不使用绝对值（显示原始 r_t 值），默认：使用绝对值

**示例**：
```bash
# 显示原始 r_t 值（可能为负）
python3 r_t_t.py --input data.json --no-abs

# 显示绝对值（默认）
python3 r_t_t.py --input data.json
```

#### 输出选项

- `--output` / `-o`: 输出文件路径，默认：自动生成

**示例**：
```bash
# 指定输出文件
python3 r_t_t.py --input data.json --output my_plot.png

# 自动生成文件名（格式：r_t_over_time_round_{round_idx}_t{start}_{end}.png）
python3 r_t_t.py --input data.json
```

#### 图表样式

- `--figsize`: 图表大小（宽,高），默认：14,8
- `--dpi`: 分辨率，默认：200
- `--marker-size`: 标记大小，默认：2
- `--linewidth`: 线宽，默认：1.5

**示例**：
```bash
# 自定义图表大小和分辨率
python3 r_t_t.py --input data.json --figsize 16,10 --dpi 300

# 自定义标记和线宽
python3 r_t_t.py --input data.json --marker-size 3 --linewidth 2.0
```

---

## 使用示例

### 示例1：查看前 100 个时间步的所有策略

```bash
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 0 --end 100
```

### 示例2：查看中间部分（500-800）的特定策略

```bash
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 500 --end 800 --strategies ga gal pretrain
```

### 示例3：查看最后 200 个时间步，显示原始值

```bash
python3 r_t_t.py --input round_2_rt_over_time_data.json --start 800 --end 1000 --no-abs
```

### 示例4：高分辨率输出，自定义样式

```bash
python3 r_t_t.py \
  --input round_2_rt_over_time_data.json \
  --start 0 --end 1000 \
  --strategies ga gal pretrain finetuned \
  --output high_res_plot.png \
  --figsize 20,12 \
  --dpi 300 \
  --marker-size 3 \
  --linewidth 2.0
```

---

## 数据文件格式

脚本期望的 JSON 数据文件格式：

```json
{
  "round_idx": 2,
  "round_title": "Tester Param Group #2",
  "max_time_steps": 1000,
  "strategies": {
    "ga": [
      {"time_step": 0, "r_t": 0.0},
      {"time_step": 1, "r_t": 0.0},
      {"time_step": 2, "r_t": -2.0},
      ...
    ],
    "gal": [
      {"time_step": 0, "r_t": 0.0},
      {"time_step": 1, "r_t": 0.0},
      ...
    ],
    ...
  }
}
```

---

## 常见问题

### Q1: 如何只显示部分时间范围？

使用 `--start` 和 `--end` 参数：

```bash
python3 r_t_t.py --input data.json --start 100 --end 500
```

### Q2: 如何只显示特定策略？

使用 `--strategies` 参数：

```bash
python3 r_t_t.py --input data.json --strategies ga gal
```

### Q3: 如何显示原始 r_t 值（包括负值）？

使用 `--no-abs` 参数：

```bash
python3 r_t_t.py --input data.json --no-abs
```

### Q4: 如何自定义输出文件名？

使用 `--output` 参数：

```bash
python3 r_t_t.py --input data.json --output my_custom_name.png
```

### Q5: 如何提高图表分辨率？

使用 `--dpi` 参数：

```bash
python3 r_t_t.py --input data.json --dpi 300
```

---

## 依赖要求

- Python 3.6+
- matplotlib
- json（标准库）

安装依赖：

```bash
pip install matplotlib
```

---

## 输出说明

### 自动生成的文件名格式

```
r_t_over_time_round_{round_idx}_t{start}_{end}.png
```

**示例**：
- `r_t_over_time_round_2_t0_1000.png`
- `r_t_over_time_round_2_t100_500.png`

### 图表内容

- **X 轴**：Time Step（时间步）
- **Y 轴**：|r_t|（绝对值）或 r_t（原始值）
- **图例**：显示各个策略的名称
- **网格**：显示网格线以便读取数值

---

## 完整参数列表

```bash
python3 r_t_t.py --help
```

---

**文档版本**：v1.0  
**最后更新**：2025年1月

