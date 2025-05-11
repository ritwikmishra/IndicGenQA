# IndicGenQA

This repository contains code for the paper:  
[**Long-context Non-factoid Question Answering in Indic Languages**](https://arxiv.org/abs/2504.13615)

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Step-by-Step Usage](#step-by-step-usage)
- [Model Checkpoint Mapping](#model-checkpoint-mapping)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Directory Structure

Your data directory should be organized as follows:

```
data/
├── dataset/
│   └── MuNfQuAD_v2.pkl
├── train_urls.pkl
├── gen2oie_models/
│   └── [downloaded Gen2OIE models here]
```

- `MuNfQuAD_v2.pkl` and `train_urls.pkl` can be obtained from [MuNfQuAD GitHub](https://github.com/ritwikmishra/MuNfQuAD).
- Gen2OIE models should be placed in `data/gen2oie_models/` (see [moie GitHub](https://github.com/dair-iitd/moie)).
- Chunking scripts are from [IndIE](https://github.com/ritwikmishra/IndIE).
- All project artefacts can be downloaded from [here](drive-link).

**Chunking model:**  
Store the chunking model at:  
```
chunking/state_dicts/model/26_repeat4_best.pth.tar
```

---


## Installation

Install the required dependencies:

```bash
pip install -r indicgenqa_req.txt
```

---

## Pipeline Overview

The pipeline consists of the following main steps:

1. **Coreference Resolution**
2. **Open Information Extraction (OIE)**
3. **Data Augmentation with Coref Data**
4. **Question Similarity Modeling**
5. **LLM Data Preparation**
6. **LLM Fine-tuning**
7. **Inference**
8. **Scoring**
9. **Score Aggregation**
10. **LLM as Judge**
11. **XAI Analysis**

---

## Step-by-Step Usage

> **Tip:** Please read the first few lines of each script for command-line arguments (args) and usage details.

### 1. Coreference Resolution

Run the coreference resolution model on the dataset (full or subset):

```bash
python 01_run_coref.py
```

---

### 2. Open Information Extraction (OIE)

- **2a. Gen2OIE on selected languages:**
    ```bash
    python 02a_run_Gen2OIE.py
    ```
- **2b. IndIE on selected languages:**
    ```bash
    python 02b_run_IndIE.py
    ```
- **2c. Intersection of OIE outputs:** We do this so that we consider only those instances where both OIE models agree on the possibility of extraction.
    ```bash
    python 02c_intersection.py
    ```

---

### 3. Data Augmentation with Coref Data

- **3a. Add coref data to OIE output (for each OIE model):**
    ```bash
    python 03a_add_res_data_to_OIE.py
    ```
- **3b. Add coref data to paragraphs (for IndIE only):**
    ```bash
    python 03b_add_res_data_to_paras.py
    ```

---

### 4. Question Similarity Modeling

- **4a. On paragraphs (Approach A1, Setting=5):**
    ```bash
    python 04a_Q_sim_para.py
    ```
- **4b. On triples without coref (Approach A2, Setting=2.1):** This is for each triple. The code supports a different setting of 2.2 which is for triples obtained from each sentence. But we did not use this seeting since Gen2OIE method does not gives triple for each sentence.
    ```bash
    python 04b_Q_sim_triples_wo_c.py
    ```
- **4c. On triples with coref (Approach A4, Setting=4):**
    ```bash
    python 04c_Q_sim_triples_w_c.py
    ```
- **4d. On paragraphs using BM25 (Ablation, Setting=5_6):**
    ```bash
    python 04d_Q_sim_para_bm25.py
    ```
- **4e. On paragraphs using LangChain retriever (Ablation, Setting=5_5):**
    ```bash
    python 04e_Q_sim_para_lc.py
    ```

> **Note:**  
> - APS model does not need to be rerun for A3 (Setting=6); use outputs from 4a and 01.
> - Baseline (B) from the paper is Setting=0.

---

### 5. LLM Data Preparation

Create data in the format required by LLMs (for all or selected settings): the setting strings specified in step 4 are used here. The data is created for all the settings. You can choose to create data for only the settings you want to run.

```bash
python 05_data_creator.py
```

---

### 6. Fine-tuning LLMs

Fine-tune LLMs using the data from step 5. Supports single/multi-GPU:

```bash
python 06_ft_LLMs.py
```

> **Note:**  
> The code is written in such a way that it can be run on multiple GPUs. So, you can run it on a single GPU or multiple GPUs. You can also choose to run it on a single GPU based on the resources available to you. The code will automatically detect the number of GPUs available and run the training accordingly. Also note that finetuning of LLMs is not required for the results that are reported on base LLMs. We have provided the code for finetuning the LLMs for your convenience. You can choose to finetune the LLMs or not based on your requirements.

---

### 7. Inference

- **7a. Inference using LLMs (base or fine-tuned, use `--checkpoint` flag):** This code supports running inferences on fine-tuned LLMs and base LLMs. You can choose to run it on fine-tuned LLMs or base LLMs based on ```--checkpoint``` flag.
    ```bash
    python 07a_inference_LLM.py
    ```
- **7b. Inference using ChatGPT (for comparison):** In the paper we report that ChatGPT performed better than all base LLMs. That is why we used it as a judge. This code runs inferences using ChatGPT if anyone wants to confirm these findings.
    ```bash
    python 07b_inference_chatgpt.py
    ```

---

### 8. Scoring

- **8a. Scoring with STS:**
    ```bash
    python 08a_sts_scores.py
    ```
- **8b. Scoring with STS using ChatGPT:** This is for 7b.
    ```bash
    python 08b_sts_scores_chatgpt.py
    ```
- **8c. Scoring with ROUGE:** We use the ROUGE model to get the scores for the results obtained from the LLMs. This code supports 7b also. 
    ```bash
    python 08c_rouge_scores.py
    ```

---

### 9. Score Aggregation

- **9a. Aggregate STS scores:**
    ```bash
    python 09a_agg_sts_scores.py
    ```
- **9b. Aggregate ROUGE scores:**
    ```bash
    python 09b_agg_rouge_scores.py
    ```

---

### 10. LLM as Judge

- **10a. Use LLMs as judges:**
    ```bash
    python 10a_LLM_as_judge.py
    ```
- **10b. Compare two experiment IDs:**
    ```bash
    python 10b_compare.py
    ```

---

### 11. XAI Analysis

- **11a. Calculate XAI scores using Ferret:** We calculate XAI scores using Ferret library. This was done to inspect how APS model is working. 
    ```bash
    python 11a_XAI_scores.py
    ```
- **11b. Construct images from SHAP/LIME values:** We constructs images from the shap and lime values from the ferret_xai json files.
    ```bash
    python 11b_images.py
    ```
- **11c. Analyze LIME/SHAP scores:** We make an analysis of lime and shape scores from ferrit_xai json files. Based on my hypothesis, "Higher intensity is given to all words when model predicts values close to 1".
    ```bash
    python 11c_analysis.py
    ```

---

## Model Checkpoint Mapping

This would be useful to run inferences with checkpoint.

| Setting         | gemma-2b | gemma-7b | llama 3.1 |
|-----------------|----------|----------|-----------|
| 0               | gn3      | gn5      | gn9       |
| 2.1 (IndIE)     | gn10     | gn11     | gn12      |
| 2.2 (Gen2OIE)   | gn31     | gn32     | gn33      |
| 4 (IndIE)       | gn19     | gn20     | gn21      |
| 4 (Gen2OIE)     | gn48     | gn49     | gn50      |
| 5               | gn22     | gn23     | gn24      |
| 5_5             | gn58     | gn59     | gn60      |
| 5_6             | gn61     | gn62     | gn63      |
| 6               | gn25     | gn26     | gn27      |

---

## Citation

If you use this code or data, please cite our paper:

```
@article{mishra2025long,
  title={Long-context Non-factoid Question Answering in Indic Languages},
  author={Mishra, Ritwik and Shah, Rajiv Ratn and Kumaraguru, Ponnurangam},
  journal={arXiv preprint arXiv:2504.13615},
  year={2025}
}

```

---

## Acknowledgements

- [MuNfQuAD](https://github.com/ritwikmishra/MuNfQuAD)
- [Gen2OIE/moie](https://github.com/dair-iitd/moie)
- [IndIE](https://github.com/ritwikmishra/IndIE)
- [Ferret XAI](https://github.com/ferret-xai/ferret)

