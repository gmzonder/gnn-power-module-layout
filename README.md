# GNN Encoders for Power Module Layout Optimization

**Investigations into the Application of Graph Neural Network Encoder for Component Placement in the Context of Power Module Layout Optimization**

Master's Thesis · M.Sc. Data Science · Kiel University of Applied Sciences · 2026
Author: **Gamze Önder**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-3C2179)
![Status](https://img.shields.io/badge/code-not%20public-lightgrey)

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
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Limitations and Future Work](#limitations-and-future-work)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Citation](#citation)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Overview

Learning-based placement methods, especially deep reinforcement learning (DRL) approaches such as the chip-placement work of Mirhoseini et al. [[5]](#references) ([Circuit Training / AlphaChip](https://github.com/google-research/circuit_training)), have shown strong results in VLSI design. Graph neural network encoders have also been used for layout planning in other domains [[6]](#references). The performance of these methods depends heavily on the **state encoder**: the network that turns the current layout into a compact representation the agent can act on.

Power module layouts are harder than VLSI placement in one important way. Their quality depends on **coupled thermal and electrical effects**, not only on geometric objectives such as wirelength [[1–3]](#references). This thesis asks whether graph-based encoders designed for VLSI placement can be adapted to **2D SiC power module layouts** and learn physically meaningful representations of them.

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

The main encoder is a **GINE** (Graph Isomorphism Network with Edge features) network [[7, 8]](#references), which uses edge attributes directly in message passing. GraphSAGE [[9]](#references) is used as a baseline.

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

$$z_i = \left[ g_{\mathrm{node}},\ e_{\mathrm{mean}},\ h_{\mathrm{cur}},\ m_{\mathrm{emb}} \right]$$

These are the pooled node representation, an edge-level summary, the embedding of the currently active component, and the encoded layout metadata. The regression head is used only during supervised pretraining. In a future RL setting it is removed, and $z_i$ becomes the agent's state representation.

<details>
<summary><b>Code excerpt: building the state embedding (simplified)</b></summary>

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_add_pool
from torch_scatter import scatter_mean


class GINEStateEncoder(nn.Module):
    """Simplified version of the thesis encoder: layout graph -> state embedding z."""

    def __init__(self, node_in=15, edge_in=7, meta_in=9, hidden=64, num_layers=2):
        super().__init__()
        self.node_encoder = nn.Linear(node_in, hidden)
        self.edge_encoder = nn.Linear(edge_in, hidden)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                    nn.ReLU(), nn.Linear(hidden, hidden),
                ),
                edge_dim=hidden,
            )
            for _ in range(num_layers)
        )
        self.edge_mlp = nn.Sequential(nn.Linear(2 * hidden + edge_in, hidden), nn.ReLU())
        self.meta_encoder = nn.Sequential(nn.Linear(meta_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        # Edge-aware message passing
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index, edge_attr))

        # 1) Pooled node representation
        g_node = global_add_pool(x, data.batch)

        # 2) Edge-level summary
        src, dst = data.edge_index
        e = self.edge_mlp(torch.cat([x[src], x[dst], data.edge_attr], dim=-1))
        e_mean = scatter_mean(e, data.batch[src], dim=0)

        # 3) Embedding of the currently active component
        h_cur = x[data.current_macro_idx + data.ptr[:-1]]

        # 4) Encoded layout metadata
        m_emb = self.meta_encoder(data.meta)

        return torch.cat([g_node, e_mean, h_cur, m_emb], dim=-1)  # state embedding z
```

> ⚠️ This is a shortened illustration of the architecture described above. The full implementation (graph construction from simulation data, feature schema, training pipeline, HPO and evaluation) is not publicly available.

</details>

### Training

- Graph-level regression, single-target and multi-target
- Adam optimizer [[11]](#references), learning-rate scheduler, early stopping on validation MAE
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

$$y_i = \left( D_{\mathrm{lin,norm}},\ F_{TH},\ F_{TIH},\ L_{\mathrm{loop}} \right)$$

**HPO search space:** hidden dimension {16, 32, 64, 128} · GINE layers {1–4} · dropout {0–0.4} · pooling {add, mean, max} · loss {MAE, MSE, Huber} · batch size · learning-rate factor.

## Key Results

- **Graph encoders transfer to power modules.** An edge-aware GINE encoder learns meaningful relationships between layout graphs and selected thermal and electrical targets.
- **Loop inductance is captured very well.** $L_{\mathrm{loop}}$ reached a **validation R² of 0.994–0.999** and a **test R² of 0.88–0.90**, depending on the normalization.
- **One shared embedding for several objectives.** The multi-target model predicts reliability, thermal and electrical metrics at the same time.
- **Normalization matters.** Z-score normalization was the most consistent. Min-max compressed predictions into narrow bands when the data did not cover the full target range.
- **Rotation robustness must be learned.** A model trained on one orientation formed separate prediction clusters for rotated layouts. Adding rotated variants to training fixed this for an unseen 90° orientation.
- **Consistent model capacity.** The best HPO configurations all used a **hidden dimension of 64, 2 GINE layers and MAE loss**. They differed in pooling and dropout.

**Best HPO configuration per normalization method (validation):**

| Normalization | Val MAE | Val RMSE | Val R² | Pooling | Hidden | Layers | Dropout | Loss |
|---|---|---|---|---|---|---|---|---|
| None | 0.0828 | 0.1217 | 0.841 | add | 64 | 2 | 0.0 | MAE |
| Z-score | 0.1178 | 0.1762 | **0.873** | max | 64 | 2 | 0.3 | MAE |
| Min-max | 0.1277 | 0.2153 | 0.868 | mean | 64 | 2 | 0.0 | MAE |

**Per-target R² of the best configurations (train / validation / test):**

| Target | None | Z-score | Min-max |
|---|---|---|---|
| $D_{\mathrm{lin,norm}}$ (reliability) | 0.929 / 0.821 / 0.535 | 0.953 / 0.831 / **0.564** | 0.932 / 0.820 / 0.131 |
| $F_{TH}$ (thermal homogeneity) | 0.843 / 0.712 / 0.617 | 0.944 / 0.813 / **0.703** | 0.931 / 0.805 / 0.052 |
| $F_{TIH}$ (thermal interaction) | 0.850 / 0.711 / 0.561 | 0.937 / 0.803 / **0.817** | 0.927 / 0.814 / 0.506 |
| $L_{\mathrm{loop}}$ (loop inductance) | 1.000 / 0.999 / 0.875 | 0.999 / 0.995 / 0.880 | 0.998 / 0.994 / **0.901** |

<!-- TODO: optionally add a prediction-vs-ground-truth figure (e.g. figures/pred_vs_gt.png) once approved -->

## Discussion

**Target learnability depends on the data, not only on the model.** A target can only be learned if the graph contains the information that describes it, if its values are available consistently, and if the dataset covers a representative range of layouts. Target selection is therefore part of the representation-learning problem. The selected target set reflects the current data and graph representation. It is not a general target set for all power module layouts.

**Absolute temperatures are not layout-only targets.** $T_{\mathrm{avg}}$, $T_{\mathrm{max}}$, $T_{\mathrm{rms}}$ and $\Delta T$ also depend on thermal simulation settings such as power loss, ambient temperature and convection coefficients. These settings are not part of the graph input, so these targets cannot be predicted reliably when the boundary conditions change. The relative thermal metrics $F_{TH}$ and $F_{TIH}$ do not have this problem and remain learnable.

**The shared embedding works, but not equally well for every target.** $L_{\mathrm{loop}}$ has consistently high test R² for every normalization method. Among the thermal objectives, $F_{TIH}$ performs best with z-score normalization. $D_{\mathrm{lin,norm}}$ and $F_{TH}$ are more sensitive to the normalization choice, and with min-max their test performance drops sharply.

**No single configuration is best for all targets.** $L_{\mathrm{loop}}$ benefits from add pooling, which suggests that accumulated graph information matters for this electrical target. $F_{TIH}$ works better with mean pooling, which fits a relative thermal metric. Normalization, pooling and regularization interact, so they should be chosen together with the target set and the data.

**Recommendation for a future RL agent.** Initial reinforcement learning experiments could start with a reduced objective set: **$L_{\mathrm{loop}}$ as the electrical objective and $F_{TIH}$ as the thermal objective**. This follows the idea of compact placement-quality objectives, such as wirelength and congestion in chip placement [[5]](#references). In the RL setting the regression head would be removed. The trained encoder would provide the state, and the selected objectives could help define the reward.

## Conclusion

**RQ1 – Transferability:** ✅ **Yes.** An edge-aware GINE encoder, inspired by VLSI placement methods, can be adapted to power module layout graphs. It learns representations that predict selected thermal and electrical performance indicators.

**RQ2 – Representation capability:** ✅ **Yes, with limits.** The learned embeddings contain information about physically relevant layout properties. For $L_{\mathrm{loop}}$ this information is very strong. This makes the encoder a promising candidate for an RL state representation.

Overall, the research questions are answered **positively but with limitations**. The thesis does not prove that the representation is already sufficient for closed-loop reinforcement learning, because no policy network, reward function, placement environment or PPO training was implemented. The encoder developed here should be seen as a **first step toward an RL-based optimization framework for power electronic module layouts**.

## Limitations and Future Work

- **Absolute temperature targets** need thermal boundary conditions (power loss, ambient temperature, convection) as part of the graph input.
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

Python · PyTorch · PyTorch Geometric [[10]](#references) · NumPy · pandas · scikit-learn · Matplotlib · Jupyter

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

## References

A selection of the main references. The full bibliography is in the thesis.

1. R. Rassmann, Y. Shen, X. Dong, U. Schuemann, and R. Mallwitz, "Optimization of semiconductor bare die positions within multi-chip power modules," in *PCIM Conference 2025*, VDE, 2025, pp. 1830–1838.
2. R. Rassmann et al., "Automated electrical and thermal optimization of conventional multi-chip power modules," in *CIPS 2026; 14th International Conference on Integrated Power Electronics Systems*, 2026, pp. 749–757.
3. R. Rassmann et al., "A holistic optimization approach for ANPC power modules in high-performance applications," in *CIPS 2026*, 2026, pp. 342–350.
4. D. S. Lopera, L. Servadei, G. N. Kiprit, S. Hazra, R. Wille, and W. Ecker, "A survey of graph neural networks for electronic design automation," in *2021 ACM/IEEE 3rd Workshop on Machine Learning for CAD (MLCAD)*, IEEE, 2021.
5. A. Mirhoseini et al., "A graph placement methodology for fast chip design," *Nature*, vol. 594, no. 7862, pp. 207–212, 2021.
6. L. Kaven, A. Göppert, and R. H. Schmitt, "Graph neural network encoder for layout planning and scheduling in line-less mobile assembly systems," *Procedia CIRP*, vol. 120, pp. 63–68, 2023.
7. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in *ICLR*, 2019.
8. W. Hu et al., "Strategies for pre-training graph neural networks," *arXiv:1905.12265*, 2019.
9. W. Hamilton, Z. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *NeurIPS*, vol. 30, 2017.
10. M. Fey and J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," *arXiv:1903.02428*, 2019.
11. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," *arXiv:1412.6980*, 2014.

## Acknowledgements

I thank my supervisors **Prof. Dr. Patrick Hennig** and **Prof. Dr. Ulf Schümann**, and my advisor **Rando Raßmann (M.Eng.)**, for their guidance and support throughout this thesis.

## Contact

Questions about the thesis are welcome. Please reach out via my [GitHub profile](https://github.com/gmzonder).
