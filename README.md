# ENTITY ALIGNMENT AND STRUCTURAL PERTURBATION FOR COMMONSENSE KNOWLEDGE GRAPH REASONING

This repository implements a **semantic alignment–enhanced graph representation learning framework** for **Commonsense Knowledge Graph Completion (CKGC)**. The pipeline integrates **BERT-based entity semantic alignment**, **pretrained language model (PLM) feature extraction**, and **graph neural network–based representation learning and decoding**.

The overall workflow follows three sequential stages:

1. **Entity semantic alignment with LLM/PLM**  
2. **Entity embedding construction via PLM fine-tuning**  
3. **Graph representation learning and link prediction decoding**

---

## Project Structure

```
.
├── Reference.txt                     # Reference information or notes
├── data_loader.py                    # Dataset loading and preprocessing
├── decoder.py                        # Decoder models (e.g., ConvTransE)
├── encoder.py                        # Graph encoders (e.g., RWGCN)
├── evaluation_utils.py               # Evaluation metrics (MRR, Hits@K)
├── evaluation_utils_singledirection.py
├── graph_class.py                    # Graph construction utilities
├── llm_alignmen.py                   # Entity semantic alignment stage
├── main.py                           # Graph training and evaluation entry
├── model.py                          # Integrated model definitions
├── parse_args.py                     # Argument parsing
├── plm_fine.py                       # PLM-based entity embedding extraction
├── utils.py                          # General utility functions
```

---

## Method Overview

### 1. Semantic Alignment Phase

In the **semantic alignment phase**, entity nodes are aligned in the semantic space using a **BERT-large** pretrained language model. This step mitigates semantic inconsistency between symbolic graph entities and textual descriptions, and prepares high-quality semantic features for downstream graph learning.

- **Model**: BERT-large  
- **Batch size**: 128  
- **Epochs**: 3  
- **Optimizer**: Adam  
- **Learning rates**:
  - BERT: `1e-4`
  - MLP: `5e-5`

---

### 2. PLM-based Entity Embedding Extraction

After alignment, the aligned entities are passed through a **PLM fine-tuning module** to generate fixed entity embeddings. These embeddings are later fused with structural and lexical features for graph representation learning.

---

### 3. Graph Representation Learning and Decoding

The graph learning and decoding stages are implemented using **PyTorch** and **DGL**. A graph neural network encoder is used to learn structural representations, followed by a convolution-based decoder (e.g., ConvTransE) for link prediction.

- **Embedding dimension**: 500  
- **Decoder convolution kernel**: 3×3 (default), configurable  
- **Initial learning rate**: 0.0003  
- **Batch size**: 1024 (graph-level)

During training:
- **MRR** is evaluated on the validation set every **15 epochs**
- **Early stopping** is applied based on validation performance

---

## Hardware Environment

Experiments are conducted on a server equipped with:

- **4 × NVIDIA RTX 3090 GPUs**

---

## Usage

### Step 1: Entity Semantic Alignment

First, perform entity semantic alignment:

```bash
python llm_alignmen.py
```

This step produces aligned semantic representations for entity nodes.

---

### Step 2: PLM-based Entity Embedding Generation

Next, generate entity embeddings using the pretrained language model:

```bash
python plm_fine.py
```

The output embeddings will be used as semantic features in graph learning.

---

### Step 3: Graph Training and Evaluation

Run the main training and evaluation pipeline:

```bash
python main.py \
  --evaluate_every=15 \
  --decoder_embedding_dim=500 \
  --decoder_batch_size=256 \
  --n_epoch=100 \
  --decoder=ConvTransE \
  --patient=10 \
  --seed=1234 \
  --regularization=1e-20 \
  --dropout=0.25 \
  --input_dropout=0.25 \
  --feature_map_dropout=0.25 \
  --lr=0.0003 \
  --bert_feat_dim=1024 \
  --optimizer=Adam \
  --dec_kernel_size=5 \
  --dec_channels=300 \
  --encoder=RWGCN_NET \
  --graph_batch_size=50000 \
  --entity_feat_dim=1324 \
  --fasttext_feat_dim=300 \
  --gnn_dropout=0.2 \
  --n_ontology=5 \
  --dynamic_graph_ee_epochs=100 \
  --start_dynamic_graph=50 \
  --rel_regularization=0.1
```

---

## Evaluation

- **Primary metric**: Mean Reciprocal Rank (MRR)  
- **Secondary metrics**: Hits@K (if enabled)  
- Validation is performed every 15 epochs

---

## Dependencies

- Python ≥ 3.8
- PyTorch
- DGL
- Transformers (HuggingFace)
- NumPy
- tqdm

---

## Notes

- Please ensure that the alignment and PLM embedding steps are executed **before** running `main.py`.
- Random seeds are fixed for reproducibility.

---

## Citation

If you use this codebase in your research, please cite the corresponding paper.
