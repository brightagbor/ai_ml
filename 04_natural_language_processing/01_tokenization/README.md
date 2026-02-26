# 🧪 NLP Implementation Lab

## Topic: Tokenization & Vocabulary Construction in TensorFlow/Keras

---

## 🎯 AIM

To implement and analyze different tokenization strategies — **word-level, subword-level, character-level, and byte-level** — using **TensorFlow/Keras**, and to understand how vocabularies are constructed and mapped to numerical representations for deep learning models.

---

## 🎯 OBJECTIVES

By the end of this lab, you should be able to:

1. Explain what tokenization is and why it is required in NLP.
2. Implement **Word-level tokenization** using `tf.keras.layers.TextVectorization`.
3. Implement **Character-level tokenization** using TensorFlow.
4. Implement **Subword tokenization** using:

   * `TextVectorization` (ngrams)
   * `tensorflow_text` (WordPiece / SentencePiece style)
5. Implement **Byte-level tokenization**.
6. Build vocabularies:

   * Word-level vocabulary
   * Character-level vocabulary
   * Subword vocabulary
   * Byte-level vocabulary
7. Convert tokens → integer IDs.
8. Visualize and compare:

   * Vocabulary size
   * OOV handling
   * Sequence length differences
9. Understand how tokenization affects model performance and generalization.


## 📚 PRE-REQUISITES

Since you already know deep learning fundamentals, you should have:

### Theoretical

* Basic understanding of:

  * NLP pipeline
  * Sequences & embeddings
  * Out-of-Vocabulary (OOV) problem
  * Softmax classification
* Understanding of:

  * Embedding layers
  * Sequence padding

### Technical

* Python
* TensorFlow 2.x
* NumPy
* Jupyter Notebook / VS Code

### Installation

```bash
pip install tensorflow
pip install tensorflow-text
```

---

# 🧠 WHAT YOU WILL ACTUALLY SEE

You said:

> I want to see how tokenization works

So in this lab you will:

* Inspect raw text
* See how it is split
* See vocabulary building
* See token → index mapping
* See padding
* Compare sequence lengths
* Analyze vocabulary growth

This will be **experimental and analytical**, not just API usage.

---

# 🔬 LAB STRUCTURE (Implementation Flow)

---

## 🔹 PART 1 — Word-Level Tokenization (Keras TextVectorization)

### Concept

Each word becomes a token.

Example:

```
"I love NLP"
→ ["I", "love", "NLP"]
```

### Vocabulary Type:

Word-level vocabulary

### What You Will Observe:

* Vocabulary size grows quickly
* OOV token appears
* Rare words are separate tokens

---

## 🔹 PART 2 — Character-Level Tokenization

### Concept

Each character becomes a token.

Example:

```
"I love"
→ ["I", " ", "l", "o", "v", "e"]
```

### Vocabulary Type:

Character-level vocabulary

### What You Will Observe:

* Very small vocabulary
* Long sequences
* No OOV problem (almost none)

---

## 🔹 PART 3 — Subword Tokenization

We implement:

### (A) N-gram Subwords (via TextVectorization)

Example:

```
playing → play + ing
```

### (B) WordPiece / SentencePiece (via tensorflow_text)

### What You Will Observe:

* Medium vocabulary size
* Handles rare words better
* Used in Transformers (BERT, etc.)

---

## 🔹 PART 4 — Byte-Level Tokenization

### Concept

Each character is encoded into UTF-8 bytes.

Example:

```
"A" → [65]
```

### Vocabulary Size:

256 possible byte values

### What You Will Observe:

* No OOV at all
* Works for multilingual text
* Used in GPT-2 style models

---

# 📊 COMPARISON TABLE (You Will Build This in Lab)

| Type      | Vocab Size | OOV Handling | Sequence Length | Used In         |
| --------- | ---------- | ------------ | --------------- | --------------- |
| Word      | Large      | Yes          | Short           | Traditional NLP |
| Character | Small      | Rare         | Long            | Char-RNN        |
| Subword   | Medium     | Rare         | Medium          | BERT, T5        |
| Byte      | 256        | None         | Long            | GPT-2           |

---

# 🧠 WHY THIS LAB IS IMPORTANT FOR YOU

Considering your research direction (deep learning & domain adaptation):

* Tokenization affects domain shift
* Subword models generalize better
* Vocabulary alignment is crucial in transfer learning
* OOV handling affects unseen domain performance

If you later implement:

* Transformers
* Large Language Models
* Cross-domain NLP models

This knowledge becomes foundational.

