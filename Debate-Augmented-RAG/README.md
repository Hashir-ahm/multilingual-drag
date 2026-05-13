<div align="center">

# Multilingual DRAG: Extending Debate-Augmented RAG to Non-English Questions

**An improvement over the ACL 2025 paper:**
**"Removal of Hallucination on Hallucination: Debate-Augmented RAG"**

**Group Members:**
Azhaff Khalid (22I-1895) · Hashir Ahmed (22I-1988) · Ahmad Akhtar (21I-1655)

---

**Original Paper Authors:**
[Wentao Hu](https://github.com/Huenao) ·
[Wengyu Zhang](https://wengyuzhang.com/) ·
[Yiyang Jiang](https://yyjiang.com/) ·
[Chen Jason Zhang](https://www.zhangchen.info/) ·
[Xiaoyong Wei](https://www4.comp.polyu.edu.hk/~x1wei/) ·
[Qing Li](https://www4.comp.polyu.edu.hk/~csqli/)

The Hong Kong Polytechnic University · Sichuan University

[![arXiv](http://img.shields.io/badge/arXiv-2505.18581-B31B1B.svg)](https://arxiv.org/abs/2505.18581)
[![ACL 2025](https://img.shields.io/badge/ACL_2025-Long_Paper-blue)](https://aclanthology.org/2025.acl-long.770/)

</div>

---

## What is DRAG?

**DRAG (Debate-Augmented RAG)** is a training-free framework introduced at ACL 2025 that reduces hallucinations in Retrieval-Augmented Generation systems. Standard RAG pipelines can actually *worsen* hallucinations when retrieved documents are biased or incorrect — the authors call this "Hallucination on Hallucination." DRAG fixes this by running structured multi-agent debates at both the retrieval and generation stages, dynamically refining queries and cross-checking answers between a proponent, a challenger, and a judge LLM.

The original implementation supports **English only** and runs a fixed number of debate rounds (r=3).

---

## Our Improvements

This repository extends DRAG with three additions:

**1. Automatic Language Detection**
Using `langdetect`, the system identifies the language of any incoming question before processing begins. No manual configuration needed.

**2. Neural Machine Translation (Multilingual Pipeline)**
Non-English questions are translated to English via Helsinki-NLP MarianMT models, run through the full DRAG pipeline, and the English answer is translated back to the original language. The pipeline supports French, German, Spanish, Arabic, Urdu, Chinese, Hindi, Russian, Turkish, and Italian — with easy extension to any language pair available through Helsinki-NLP.

```
[Non-English Question]
        ↓  detect language
        ↓  translate → English
  [English Question]
        ↓  Retrieval Debate  (proponent / challenger / judge)
        ↓  Response Debate   (proponent / challenger / judge)
  [English Answer]
        ↓  translate → original language
[Non-English Answer]
```

No retraining. No changes to DRAG internals. A pure drop-in extension.

**3. Adaptive Debate Stopping**
The original paper always runs exactly 3 retrieval and 3 response debate rounds. We replace this with a confidence-based early stopping mechanism. After each round, if the judge's language signals high certainty and the proponent and challenger agree, the debate halts early. This reduces LLM calls without sacrificing accuracy.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on Kaggle T4, 16 GB VRAM)
- A HuggingFace account (free) for model access

### Installation

```bash
# 1. Clone the original DRAG repository
git clone https://github.com/Huenao/Debate-Augmented-RAG.git
cd Debate-Augmented-RAG

# 2. Install FlashRAG (required by DRAG internals)
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG && pip install -e . && cd ..

# 3. Install all dependencies
pip install -r requirements.txt

# 4. For PyTorch with CUDA 11.8 (Kaggle / most Colab environments):
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** If you run into conflicts during installation, follow the [FlashRAG installation guide](https://github.com/RUC-NLPIR/FlashRAG/tree/main#wrench-installation) first, then install the multilingual packages (`langdetect`, `sentencepiece`, `sacremoses`) on top.

### Datasets

Datasets follow the same format as [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG/tree/main#datasets) and are available on [HuggingFace](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). After downloading, place them under a `dataset/` folder:

```
Debate-Augmented-RAG/
├── dataset/
│   ├── 2wiki/
│   ├── HotpotQA/
│   ├── NQ/
│   ├── PopQA/
│   ├── StrategyQA/
│   └── TriviaQA/
```

Supported datasets: `NQ`, `TriviaQA`, `PopQA`, `2WikiMultihopQA`, `HotpotQA`, `StrategyQA`.

### Document Corpus & Index

Download the `wiki18_100w` corpus and its `e5-base-v2` index from [ModelScope](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files) and place them under `wiki_corpus/`:

```
Debate-Augmented-RAG/
├── wiki_corpus/
│   ├── e5_flat_inner.index
│   └── wiki18_100w.jsonl
```

### Model

This project was tested with **Qwen2.5-1.5B-Instruct** (public, no approval needed, fits on a free T4 GPU). It also supports any LLM compatible with HuggingFace or vLLM — set the model path in `config/base_config.yaml` under `model2path`.

```bash
# To use Qwen (recommended for free-tier GPUs):
# In config/base_config.yaml, set:
# llama3-8B-instruct: "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## Running

### Original DRAG (English only)

```bash
python main.py --method_name "DRAG" \
               --gpu_id "0" \
               --dataset_name "StrategyQA" \
               --generator_model "llama3-8B-instruct"
```

Arguments:

| Argument | Options | Description |
|---|---|---|
| `--method_name` | `DRAG`, `Naive Gen`, `Naive RAG`, `FLARE`, `Iter-RetGen`, `IRCoT`, `SuRe`, `Self-RAG`, `MAD` | RAG method to use |
| `--gpu_id` | e.g. `"0"` | GPU device ID |
| `--dataset_name` | `NQ`, `TriviaQA`, `PopQA`, `2wiki`, `HotpotQA`, `StrategyQA` | Dataset to evaluate on |
| `--generator_model` | e.g. `"llama3-8B-instruct"` | Generator LLM |
| `--max_query_debate_rounds` | integer | Retrieval debate rounds (DRAG only) |
| `--max_answer_debate_rounds` | integer | Response debate rounds (DRAG only) |

### Multilingual DRAG (our extension)

Open and run `MultilingualDRAG_improvement.ipynb` cell by cell. The notebook:

1. Installs all dependencies
2. Loads Qwen2.5-1.5B-Instruct
3. Initialises the BM25 retriever (no Wikipedia download required for the demo)
4. Runs the full multilingual pipeline on questions in English, French, German, and Spanish
5. Evaluates with Exact Match and token-level F1, and prints a per-language results table

To run a single question:

```python
result = multilingual_drag("Qui a inventé le téléphone?", verbose=True)
print(result.final_answer)   # Answer translated back to French
```

---

## Supported Languages

| Code | Language | Translation models used |
|------|----------|------------------------|
| `fr` | French | `Helsinki-NLP/opus-mt-fr-en` / `opus-mt-en-fr` |
| `de` | German | `Helsinki-NLP/opus-mt-de-en` / `opus-mt-en-de` |
| `es` | Spanish | `Helsinki-NLP/opus-mt-es-en` / `opus-mt-en-es` |
| `ar` | Arabic | `Helsinki-NLP/opus-mt-ar-en` / `opus-mt-en-ar` |
| `ur` | Urdu | `Helsinki-NLP/opus-mt-ur-en` / `opus-mt-en-ur` |
| `zh` | Chinese | `Helsinki-NLP/opus-mt-zh-en` / `opus-mt-en-zh` |
| `hi` | Hindi | `Helsinki-NLP/opus-mt-hi-en` / `opus-mt-en-hi` |
| `ru` | Russian | `Helsinki-NLP/opus-mt-ru-en` / `opus-mt-en-ru` |
| `tr` | Turkish | `Helsinki-NLP/opus-mt-tr-en` / `opus-mt-en-tr` |
| `it` | Italian | `Helsinki-NLP/opus-mt-it-en` / `opus-mt-en-it` |

Adding a new language only requires adding one entry to the `TRANSLATION_MODELS` dictionary.

---

## Evaluation

We evaluate using the same metrics as the original paper:

- **EM (Exact Match):** 1 if the normalised predicted answer matches the gold answer exactly, else 0.
- **F1:** Token-level overlap between prediction and gold answer.

The notebook produces a results table in the style of the original paper's Table 1, broken down by language, and reports how many debate rounds were used on average (to demonstrate the savings from adaptive stopping).

---

## Acknowledgements

This project is built on top of the following open-source works:

- **DRAG** — [Huenao/Debate-Augmented-RAG](https://github.com/Huenao/Debate-Augmented-RAG) (Hu et al., ACL 2025). All credit for the core debate-augmented RAG framework goes to the original authors.
- **FlashRAG** — [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). The retrieval pipeline and dataset utilities are from FlashRAG.
- **Helsinki-NLP MarianMT** — Neural machine translation models from the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) group on HuggingFace Hub.
- **Qwen2.5** — [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) by Alibaba Cloud, used as the generator LLM.

---

## Citation

If you use the original DRAG framework, please cite the original paper:

```bibtex
@inproceedings{hu-etal-2025-removal,
    title     = "Removal of Hallucination on Hallucination: Debate-Augmented {RAG}",
    author    = "Hu, Wentao and Zhang, Wengyu and Jiang, Yiyang and
                 Zhang, Chen Jason and Wei, Xiaoyong and Qing, Li",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association
                 for Computational Linguistics (Volume 1: Long Papers)",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.acl-long.770/",
    pages     = "15839--15853",
    ISBN      = "979-8-89176-251-0",
}
```
