# AgentVNE

AgentVNE: LLM-Augmented Graph Reinforcement Learning for Affinity-Aware Multi-Agent Placement in Edge Agentic AI

## Project Overview

AgentVNE is a virtual network embedding framework designed for agentic edge computing scenarios. It employs a dual-layer architecture that combines semantic perception with graph similarity learning to address the deployment challenges of dynamic workflows on heterogeneous edge infrastructure.

**Key Advantages:**
- **Semantic Perception**: Identifies implicit semantic constraints of workflow nodes through LLM
- **Graph Similarity Learning**: Pre-training + PPO fine-tuning strategy to precisely capture topological similarities
- **Dynamic Adaptation**: Supports real-time workflow arrivals and dynamic resource changes
- **Performance Improvement**: Reduces communication latency to less than 40% of baselines and improves acceptance rate by 5%-10%

## System Architecture

AgentVNE adopts a dual-layer architecture that combines semantic perception with topological reasoning:

### Layer 1: LLM-based Semantic Perception & Constraint Resolution

Performs semantic perception and constraint resolution through large language models, enabling intelligent node matching and resource augmentation.

**Core Functions:**
- 🔍 **Semantic Perception**: Analyzes prompts of virtual nodes (VN) to understand semantic requirements and functional characteristics
- 🎯 **Constraint Identification**: Automatically identifies special execution environments required by nodes (e.g., PCI DSS security environment, GPU computing environment, camera hardware, etc.)
- 🔗 **Intelligent Matching**: Matches VN nodes requiring special environments to appropriate substrate network (SN) nodes
- 📊 **Resource Augmentation**: Provides semantic-level constraints and bias information for subsequent embedding decisions

**Implementation Module:** `LLM_resource_augmentation/node_optimizer/`

This module uses LLM to analyze workflow node prompts, determines whether nodes require special execution environments, and automatically matches them to appropriate SN nodes, providing semantic constraints for Layer 2 embedding decisions.

**Usage Example:**
```bash
cd LLM_resource_augmentation/node_optimizer
uv run run_optimizer.py
```

### Layer 2: Graph Similarity Deep Embedding & Policy Optimization

Deep embedding and policy optimization layer based on graph neural networks and reinforcement learning.

**Core Functions:**
- 🧠 **Graph Encoding**: Uses GCN encoders to process VN and SN graphs separately, extracting node features
- 🔄 **Transformer Enhancement**: Enhances node feature representations through Transformer Encoder
- 🎯 **Similarity Computation**: Uses Neural Tensor Network (NTN) to compute matching probabilities between VN nodes and SN nodes
- 🚀 **PPO Optimization**: Fine-tunes policy network in real environments using Proximal Policy Optimization (PPO) algorithm

**Training Pipeline:**
1. **Pre-training Phase**: Supervised learning using NodeRank labels to learn graph similarity representations
2. **Fine-tuning Phase**: Optimizes policy using PPO algorithm in dynamic environments to maximize acceptance rate and resource utilization

**Implementation Modules:**
- `model.py`: SimuVNE model (policy network)
- `pretrain.py`: Pre-training script
- `fine_tuning.py`: PPO fine-tuning script
- `env.py`: Reinforcement learning environment (SimuVNEEnv)

## Core Features

- 🎯 **Two-Stage Training**: Pre-training + PPO fine-tuning
- 🧠 **Graph Neural Networks**: GCN encoder + Transformer Encoder
- 🔄 **Reinforcement Learning**: PPO algorithm for policy optimization
- 📊 **Multi-Strategy Support**: Baseline methods including greedy, genetic algorithm, NodeRank, etc.
- 🔍 **Semantic Perception**: LLM-driven constraint identification and resource augmentation

## Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
conda activate AgentVNE
```

### Data Preparation

1. Place SN topology files in the `topo/` directory
2. Place Workflow topology files in the `Workflow_topo/` directory
3. Generate pre-training dataset:

```bash
python dataset_generate_1.py \
    --sn_topo topo/SN_topology.json \
    --workflow_topo Workflow_topo/workflow1_topo.json \
    --workflow_noderank Workflow_topo/workflow1_noderank.json \
    --output pretrain_data/pretrain_dataset.pt \
    --workflows_per_episode 10 \
    --num_episodes 50
```

### Training Pipeline

**1. Pre-training**
```bash
python pretrain.py \
    --data_path pretrain_data/pretrain_dataset.pt \
    --output_dir pretrain_outputs \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.001
```

**2. Fine-tuning**
```bash
python fine_tuning.py \
    --pretrain_model pretrain_outputs/checkpoint_latest.pt \
    --sn_topology topo/SN_topology.json \
    --workflow_types Workflow_topo/workflow1_topo.json \
    --output_dir finetuning_output \
    --num_episodes 1000 \
    --max_arrived_tasks 100
```

**3. Testing & Evaluation**
```bash
python tester.py \
    --sn_topology topo/SN_topology.json \
    --workflow workflow1=Workflow_topo/workflow1_topo.json \
    --strategy ga --strategy greedy --strategy pretrain --strategy finetuned \
    --parameter arrival_rate=0.25,mean_lifetime=40,max_time_steps=11000,seed=42 \
    --plot
```

## Project Structure

```
agentvne/
├── model.py                    # SimuVNE model (policy network)
├── model__sigmoid.py           # SimuVNE model variant (with Sigmoid)
├── env.py                      # Environment definition (SimuVNEEnv, WorkflowGenerator)
├── pretrain.py                 # Pre-training script
├── fine_tuning.py              # PPO fine-tuning script
├── dataset_generate_1.py       # Dataset generation
├── tester.py                   # Multi-strategy testing script
├── LLM_resource_augmentation/  # Layer 1: LLM semantic perception & constraint resolution
│   └── node_optimizer/         # Node optimizer (VN-SN intelligent matching)
├── baselines/                  # Baseline methods
├── topo/                       # SN topology files and tools
├── Workflow_topo/              # Workflow topology files
├── pretrain_data/              # Pre-training dataset
├── pretrain_outputs/           # Pre-training model outputs
└── finetuning_output/          # Fine-tuning model outputs
```

## Model Architecture

The **SimuVNE model** consists of the following components:

- **GCN Encoder**: Encodes VN and SN graphs to extract node embeddings
- **Transformer Encoder**: Enhances node feature representations and captures graph structure information
- **Neural Tensor Network (NTN)**: Computes matching probabilities between VN nodes and SN nodes
- **Output Layer**: Generates probability matrix [N_v, N_s], representing matching probabilities from each VN node to each SN node

**Training Strategy:**
- **Pre-training**: Learns NodeRank label distribution using MSE loss
- **Fine-tuning**: Optimizes policy using PPO algorithm with reward function based on acceptance rate and resource utilization

## Supported Strategies

- `ga`: Genetic Algorithm
- `gal-vne`: Greedy algorithm based on NodeRank
- `greedy`: Greedy algorithm based on SN sorting
- `pretrain`: Pre-trained model (ft_n)
- `finetuned`: Fine-tuned model (ft1)

## Configuration

Main configuration is in `config.json`, including model dimensions, training parameters, etc. Command-line arguments support flexible configuration of network topology, workflow types, training parameters, etc.

## License

MIT License

---

**Note: This project is under active development and APIs may change.**
