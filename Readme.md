# Qwen 2.5 3B Research QA Fine-tuning

Fine-tuned version of Qwen 2.5 3B for answering questions about AI research papers and technical documentation.

## Features

- **Synthetic Dataset Generation**: Automatically generate datasets from technical documents.
- **LoRA Fine-tuning**: Utilize Low-Rank Adaptation for parameter-efficient model fine-tuning. :contentReference[oaicite:0]{index=0}
- **4-bit GGUF Quantization**: Apply 4-bit quantization for efficient inference.
- **End-to-End Training Pipeline**: Comprehensive scripts for data processing, training, and evaluation.
- **Evaluation Metrics**: Assess model performance using ROUGE-L and exact match scores.

## Requirements

- Python 3.9+
- NVIDIA GPU (16GB+ VRAM recommended)
- Google Colab Pro (recommended for free tier users)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Qwen-Research-QA-FineTuning.git
   cd Qwen-Research-QA-FineTuning
   ```
2 Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Quick Start
Dataset Preparation: Place your research documents in data/raw_documents/ and run:

  ```bash
  python scripts/generate_dataset.py --input_dir data/raw_documents --output_dir data/synthetic_dataset
  Fine-tuning:
  ```

```bash
  python scripts/train.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --dataset_dir data/synthetic_dataset \
    --output_dir models/fine_tuned_qwen
```
## Quantization:

```bash
  python scripts/quantize.py \
      --input_dir models/fine_tuned_qwen \
      --output_dir models/quantized \
      --quant_type q4_0
```
## Inference:

```python
from scripts.inference import ResearchQA

qa_system = ResearchQA("models/quantized/qwen-3b-finetuned-Q4.gguf")
response = qa_system.ask("What is 3FS?")
print(response)
```
## Evaluation
```bash
python scripts/evaluate.py \
    --model_path models/quantized/qwen-3b-finetuned-Q4.gguf \
    --test_set data/synthetic_dataset/test.json
````

## Configuration
- Modify key parameters in scripts/config.py:

```python
# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 32
TARGET_MODULES = ["c_attn", "w1", "w2"]

# Training Parameters
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 2048
```
   
