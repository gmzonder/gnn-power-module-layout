# GNN Encoders for Power Module Layout Optimization

**Investigations into the Application of Graph Neural Network Encoder for Component Placement in the Context of Power Module Layout Optimization**

Master's Thesis · M.Sc. Data Science · Kiel University of Applied Sciences · 2026
Author: **Gamze Önder**

---

This repository gives an overview of my master's thesis. It explains the problem, method, experiments and main results, and describes how the code base is organized.

> **Code and data availability:** The simulation datasets used in this thesis are proprietary, and the implementation belongs to an ongoing research project. For these reasons, neither the code nor the data is published here. The project structure below is described for documentation purposes only.

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Contributions](#contributions)
- [Method](#method)
- [Experiments](#experiments)
- [Key Results](#key-results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

Learning-based placement methods, especially deep reinforcement learning (DRL) approaches such as the chip-placement work of [Mirhoseini et al.](https://www.nature.com/articles/s41586-021-03544-w) ([Circuit Training / AlphaChip](https://github.com/google-research/circuit_training)), have shown strong results in VLSI design. Their performance depends heavily on the **state encoder**: the network that turns the current layout into a compact representation the agent can act on.

Power module layouts are harder than VLSI placement in one important way. Their quality depends on **coupled thermal and electrical effects**, not only on geometric objectives such as wirelength. This thesis asks whether graph-based encoders designed for VLSI placement can be adapted to **2D SiC power module layouts** and learn physically meaningful representations of them.

The thesis focuses on the **encoder stage** of a future RL framework. The encoder is pretrained with supervised learning to predict simulation-derived thermal and electrical layout metrics. The full RL loop (policy and value networks, reward design, PPO training) is outside the scope of this work.

```mermaid
flowchart LR
    L["Power module layout<br/>(graph observation)"] --> E["GNN encoder<br/>(this thesis)"]
    E --> Z["State embedding z_t"]
    Z -.-> P["Policy network"]
    Z -.-> V["Value network"]
    P -.-> A["Placement action"]
    A -.-> L
    classDef future stroke-dasharray: 5 5
    class P,V,A future
```
<sub>Dashed components belong to a future RL framework and are not implemented in this thesis.</sub>

## Research Questions

1. **RQ1 – State encoding transferability.** Can Edge-GNN-based encoder architectures, originally developed for VLSI placement, be adapted to represent the multi-physics constraints and complex geometries of conventional 2D SiC power module layouts?
2. **RQ2 – Representation capability.** Can the learned graph embeddings distinguish between layouts with different physical properties, such as high and low inductance or different thermal behavior? Do they provide a suitable basis for a future DRL-based placement agent?

## Contributions

- A **graph-based representation pipeline** for power module layouts
- **Supervised training of GNN encoders** on simulation-derived thermal and electrical targets
- A systematic **experimental evaluation** of target normalization, pooling, loss functions, rotated layouts and hyperparameter configurations
- An **analysis** of whether the learned embeddings capture physically meaningful layout properties

## Method

### Pipeline

```mermaid
flowchart LR
    A["Raw layout, simulation<br/>and metadata files"] --> B["Preprocessing"]
    B --> C["Graph construction<br/>(nodes = components,<br/>edges = connections)"]
    C --> D["Feature encoding<br/>and normalization"]
    D --> E["Supervised GINE<br/>encoder training"]
    E --> F["Graph-level<br/>state embedding"]
```

### Layout as a graph

Each layout is converted into a graph:

- **Nodes** are physical components: chips, bond pads and connectors. Node features include position, size, voltage potential, normalized power loss and component type (15 encoded features).
- **Edges** are physical or electrical connections: bond wires and conductor traces. Edge features include bond length, bond-loop height, potential type and connection type (7 encoded features).
- **Metadata** holds global layout information: canvas size, grid size, number of devices and number of connectors.

<!-- TODO: add an example layout and its graph representation (thesis Fig. 3.3 / 3.4) once approved -->

### Encoder architecture

The main encoder is a **GINE** (Graph Isomorphism Network with Edge features) network, which uses edge attributes directly in message passing. GraphSAGE is used as a baseline.

```mermaid
flowchart LR
    NF["Node features"] --> NE["Node encoder"]
    EF["Edge features"] --> EE["Edge encoder"]
    NE --> G["GINEConv layers<br/>(edge-aware message passing)"]
    EE --> G
    G --> PL["Pooling<br/>(add / mean / max)"]
    G --> EM["Edge MLP<br/>→ mean edge summary"]
    G --> CM["Current-component<br/>embedding"]
    M["Metadata"] --> ME["Metadata MLP"]
    PL --> Z["Concat → state embedding z"]
    EM --> Z
    CM --> Z
    ME --> Z
    Z --> H["Regression head<br/>(pretraining only)"]
```

The final state embedding combines four parts:

$$z_i = [\,g_\text{node},\ e_\text{mean},\ h_\text{cur},\ m_\text{emb}\,]$$

These are the pooled node representation, an edge-level summary, the embedding of the currently active component, and the encoded layout metadata. The regression head is used only during supervised pretraining. In a future RL setting it is removed, and $z_i$ becomes the agent's state representation.

### Training

- Graph-level regression, single-target and multi-target
- Adam optimizer, learning-rate scheduler, early stopping on validation MAE
- Loss functions: MAE, MSE, Smooth L1 / Huber
- Target normalization: none, z-score, min-max
- Metrics (MAE, RMSE, R²) are computed on the original physical scale

## Experiments

| # | Experiment | Purpose |
|---|---|---|
| 1.1 | **Single-target prediction** | Test how learnable each target is, and the effect of normalization, pooling and loss |
| 1.2 | **Thermal boundary conditions** | Check whether temperature targets stay predictable when thermal boundary conditions vary |
| — | **Target selection** | Choose a stable, physically meaningful target vector |
| 2 | **Multi-target prediction** | Test whether one shared embedding can predict several objectives |
| 3 | **Rotated layouts** | Test robustness to layout rotation (train on 0°/180°/270°, test on unseen 90°) |
| 4 | **Hyperparameter optimization** | 300 random-search trials per normalization method, ranked by validation R² |

**Selected multi-target vector** (reliability, thermal homogeneity, thermal interaction, electrical):

$$y_i = \left(D_{\text{lin,norm}},\ F_{TH},\ F_{TIH},\ L_{\text{loop}}\right)$$

**HPO search space:** hidden dimension {16, 32, 64, 128} · GINE layers {1–4} · dropout {0–0.4} · pooling {add, mean, max} · loss {MAE, MSE, Huber} · batch size · learning-rate factor.

## Key Results

- **Graph encoders transfer to power modules.** An edge-aware GINE encoder learns meaningful relationships between layout graphs and physical targets (RQ1).
- **Loop inductance is captured very well.** $L_\text{loop}$ reached a **validation R² of 0.994–0.999** and a **test R² of 0.88–0.90**, depending on the normalization.
- **One shared embedding for several objectives.** The multi-target model predicts reliability, thermal and electrical metrics at the same time, which is what a future RL state representation needs (RQ2).
- **Normalization matters.** Z-score normalization was the most consistent. Min-max compressed predictions when the data did not cover the full target range.
- **Rotation robustness must be learned.** A model trained on one orientation formed separate prediction clusters for rotated layouts. Adding rotated variants to training fixed this for an unseen 90° orientation.
- **Consistent best configuration.** The best HPO configurations all used a **hidden dimension of 64, 2 GINE layers and MAE loss**. They differed in pooling and dropout.

**Best HPO configuration per normalization method (validation):**

| Normalization | Val MAE | Val RMSE | Val R² | Pooling | Hidden | Layers | Dropout | Loss |
|---|---|---|---|---|---|---|---|---|
| None | 0.0828 | 0.1217 | 0.841 | add | 64 | 2 | 0.0 | MAE |
| Z-score | 0.1178 | 0.1762 | **0.873** | max | 64 | 2 | 0.3 | MAE |
| Min-max | 0.1277 | 0.2153 | 0.868 | mean | 64 | 2 | 0.0 | MAE |

**Per-target test R² (best z-score configuration):**

| $D_\text{lin,norm}$ | $F_{TH}$ | $F_{TIH}$ | $L_\text{loop}$ |
|---|---|---|---|
| 0.564 | 0.703 | 0.817 | 0.880 |

<!-- TODO: optionally add a prediction-vs-ground-truth figure (e.g. figures/pred_vs_gt.png) once approved -->

## Limitations and Future Work

- **Absolute temperature targets** ($T_\text{max}$, $T_\text{avg}$, …) depend on thermal boundary conditions (power loss, ambient temperature, convection) that are not yet part of the graph input.
- **Dataset size and diversity.** Larger and more balanced datasets are needed, covering more netlist families, chip counts, rotations and boundary conditions.
- **More robust evaluation.** Netlist-based cross-validation and ablation studies of the graph representation.
- **Rotation-invariant modeling**, instead of relying on data augmentation alone.
- **Full RL integration.** Connect the pretrained encoder to policy and value networks and a PPO-based placement environment.

## Project Structure

The implementation is organized as a modular Python library plus experiment notebooks. Every experiment is fully described by a JSON config file.

```
├── lib/
│   ├── config.py                     # Loads JSON experiment configs, selects the device
│   ├── pre_processing/
│   │   ├── feature_schema.py         # Node/edge feature definitions, encoding, normalization
│   │   ├── convert_graph_to_pyg_data.py   # Raw layout graph → PyTorch Geometric Data
│   │   ├── graph_dataset_builder.py  # Joins graphs, targets and metadata into datasets
│   │   └── build_loaders_from_datasets.py # Leakage-safe normalization and DataLoaders
│   ├── models/
│   │   ├── gine.py                   # GINE-based graph-level regressor (main encoder)
│   │   ├── sage.py                   # GraphSAGE baselines
│   │   ├── edge_mlp.py               # Learned edge embeddings
│   │   ├── metadata.py               # Metadata encoder
│   │   └── model_factory.py          # Builds the model selected in the config
│   ├── training/
│   │   ├── run_experiment.py         # End-to-end experiment runner
│   │   ├── training_utils.py         # Training loop, losses, early stopping
│   │   ├── targets_normalization.py  # None / z-score / min-max target scaling
│   │   ├── k_fold.py                 # Grouped k-fold cross-validation
│   │   └── inference.py              # Prediction on new layout graphs
│   └── post_processing/
│       ├── plots.py                  # Prediction-vs-ground-truth and training curves
│       ├── tables.py                 # Metric tables (incl. LaTeX export)
│       ├── compare_runs.py           # Compares saved experiment runs
│       └── embedding_analysis.py     # Projects and analyzes learned embeddings
├── configs/                          # JSON configs for each experiment
├── experiments/                      # Training and analysis notebooks (Experiments 1–3)
└── hyperparameter_optimization/      # HPO notebooks (random search per normalization)
```

## Tech Stack

Python · PyTorch · PyTorch Geometric · NumPy · pandas · scikit-learn · Matplotlib · Jupyter

## Citation

```bibtex
@mastersthesis{oender2026gnnencoder,
  author = {Gamze {\"O}nder},
  title  = {Investigations into the Application of Graph Neural Network Encoder
            for Component Placement in the Context of Power Module Layout Optimization},
  school = {Kiel University of Applied Sciences},
  year   = {2026},
  type   = {Master's Thesis}
}
```

## Acknowledgements

I thank my supervisors **Prof. Dr. Patrick Hennig** and **Prof. Dr. Ulf Schümann**, and my advisor **Rando Raßmann (M.Eng.)**, for their guidance and support throughout this thesis.

## Contact

Questions about the thesis are welcome. Please reach out via my [GitHub profile](https://github.com/gmzonder).
