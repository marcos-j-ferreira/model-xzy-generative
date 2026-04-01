# model-xzy-generative

A simple generative text model capable of generalizing from a small dataset (53 words). Given an input, it can predict and generate the next words in a sequence.

[![Hugging Face - xzy_mini](https://img.shields.io/badge/HuggingFace-1f1f1f?style=for-the-badge&logo=huggingface&logoColor=FFD21E)](https://marcos-j-ferreira-xzy-mini.hf.space/?__theme=system&deep_link=ADUi7H0udRc)
---

## How it works

We can break the model into three main parts for clarity:

```
   Raw Text
      │
      ▼
 [ Data Processing ]
      │
      ▼
 [   Transformer   ]
      │
      ▼
 [    Training     ]
      │
      ▼
 Generated Text
```

---

## 1. Data Preparation

The model cannot understand text the way humans do, so the data must be processed before training. This step has three main stages:

```
"hello world" 
      │
      ▼
 [ Vocabulary ]
      │   → splits text into words
      ▼
 [  Tokens    ]
      │   → converts words into indices
      ▼
 [ Embedding  ]
          → maps indices into vectors
```

* **Vocabulary (vocab):** splits the input text into individual words.
* **Tokenization (tokens):** converts each word into a numerical index.
* **Embedding:** transforms these indices into vectors (embedding table), allowing the model to learn relationships between words.

---

## ## 2. Transformer

The Transformer is the core of the model. Below is a more accurate representation of its architecture:

```
Input Embeddings
       │
       ▼
+----------------------+
|   Positional Encoding|
+----------------------+
       │
       ▼
+----------------------+
|   Multi-Head         |
|   Self-Attention     |
+----------------------+
       │
       ▼
+----------------------+
|   Add & Norm         |
+----------------------+
       │
       ▼
+----------------------+
| Feed Forward Network |
+----------------------+
       │
       ▼
+----------------------+
|   Add & Norm         |
+----------------------+
       │
       ▼
      Output
```

### Key components:

* **Positional Encoding**
  Adds information about word order (since the Transformer itself has no notion of sequence).

* **Multi-Head Self-Attention**
  Allows the model to look at different parts of the sentence simultaneously and capture multiple relationships.

* **Add & Norm**
  Residual connections + normalization to stabilize training.

* **Feed Forward Network (FFN)**
  A small neural network applied to each position independently.

---

### Intuition

```
Word → looks at all other words → decides what matters → updates representation
```

Example:

```
"The cat sat on the mat"
      ↑
   "sat" attends to "cat"
```

---

### Important note

In a real implementation:

* This block is stacked multiple times (layers)
* Each layer refines the understanding of the sequence
* For generative models, masking is applied so the model cannot "see the future"


## 3. Training

During training, the model learns to predict the next word based on previous words.

```
Input:  "the cat"
Target: "sat"

Model Prediction → compares with target → adjusts weights
```

* The model receives sequences of words.
* It tries to predict the next word.
* The prediction is compared with the correct answer.
* Errors are used to update the model (via backpropagation).

Over time, the model improves its predictions.

---

## After these three processes

Once the data has been processed, passed through the Transformer, and trained:

```
Input → "the cat"
        │
        ▼
Model → "sat on the mat"
```

The model can now:

* Generate text sequences
* Predict next words
* Generalize from a small dataset

---

## Final considerations

* This is a **lightweight generative model**, designed to work even with a small dataset.
* While it does not reach the complexity of large language models, it demonstrates the core ideas behind text generation.
* Performance depends heavily on:

  * Data quality
  * Training time
  * Model architecture

---

## Summary

In short, this is a generative text model that:

* Processes text into numerical representations
* Learns patterns using a Transformer
* Generates new text based on learned context

```
Small Data → Learned Patterns → Generated Text
```
