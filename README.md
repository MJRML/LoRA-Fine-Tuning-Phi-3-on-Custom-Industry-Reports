# Fine-Tuning Phi-3 Mini with LoRA on a Crypto Compliance Report

## Project Overview
- This project demonstrates how to fine-tune the microsoft/phi-3-mini-4k-instruct Large Language Model (LLM) using LoRA (Low-Rank Adaptation) for domain adaptation. We inject new knowledge into the model using a PDF report on cryptocurrency compliance in 2025. The model is then able to answer questions based on the newly learned context — without full model retraining.

- Designed to work entirely in Google Colab using 4-bit quantization, this project is resource-efficient and perfect for extending open-source LLMs with domain-specific updates.

## Technologies & Tools
- Component	Tool/Library
- LLM	microsoft/phi-3-mini-4k-instruct
- Fine-tuning	PEFT (LoRA)
- Quantization	bitsandbytes (4-bit inference)
- Tokenization	Hugging Face transformers
- Dataset creation	Hugging Face datasets
- Training	Trainer from transformers
- PDF parsing PyPDF2 
- Runtime	Google Colab (CUDA)

## Highlights
- Efficient LoRA fine-tuning on a quantized LLM
- Domain adaptation from an unstructured PDF
- Lightweight Colab-compatible workflow
- Real-world enterprise NLP use case
