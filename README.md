# Mini LLM from Scratch using PyTorch

This project demonstrates how to build and train a **small custom Large Language Model (LLM)** from scratch using **PyTorch**.  
The model is based on a **GPT-style Transformer architecture** and is trained using **next-token prediction** on a custom text dataset.

The implementation is fully **offline**, lightweight, and suitable for **students and beginners** who want to understand how LLMs work internally.

---

## 🚀 Features

- Transformer-based language model (GPT-style)
- Multi-head self-attention with causal masking
- Positional and token embeddings
- Next-token prediction training objective
- Fully offline character-level tokenizer
- Text generation after training
- Runs on CPU or single GPU

---

## 🧠 Model Architecture

- Token Embedding
- Positional Embedding
- Transformer Blocks (Self-Attention + Feed Forward)
- Layer Normalization
- Linear Output Head

The model learns to predict the **next character** in a sequence, which is the fundamental idea behind modern LLMs.

---

## 📂 Project Structure

