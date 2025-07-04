# Ivy-Fake: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection

[![Paper](https://img.shields.io/badge/paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2506.00979)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake)
[![GitHub Code](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Pi3AI/Ivy-Fake) [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

![Intro-image](static/images/figure1-poster-v2_00.png)

**Ivy-Fake** is a novel, unified, and large-scale dataset specifically designed for explainable multimodal AI-Generated Content (AIGC) detection. It is the first large-scale benchmark for this purpose, covering both images and videos. The project also introduces **Ivy-XDetector**, a unified vision-language model that achieves state-of-the-art performance in detecting and explaining AI-generated images and videos.

The rapid advancement of AIGC has led to hyper-realistic synthetic media, raising significant concerns about authenticity and misinformation. Current detection methods often operate as black-box classifiers, lacking interpretability and unified support for different modalities. Ivy-Fake addresses these limitations by providing a comprehensive benchmark and an explainable detection framework.

## Key Features

* **Unified Multimodal Dataset:** Ivy-Fake contains over 150,000 richly annotated training samples (images and videos) and 18,700 evaluation examples.
* **Explainable Annotations:** Each sample is accompanied by detailed natural-language reasoning, going beyond simple binary labels to explain detected spatial and temporal artifacts.
* **IVY-XDETECTOR Model:** A unified vision-language architecture that performs state-of-the-art detection and provides human-readable explanations for both image and video AIGC.
* **Focus on Real-World Scenarios:** The dataset includes diverse content generated by various cutting-edge architectures (GANs, Diffusion models, Transformers) and authentic content from real-world contexts.
* **Publicly Available:** The Ivy-Fake dataset is publicly accessible on Hugging Face Datasets.

## Project Overview

The Ivy-Fake framework (illustrated below) conducts an in-depth analysis of temporal and spatial artifacts to enable explainable detection of AI-generated content.

*Caption: Overview of the IVY-FAKE framework: By conducting in-depth analysis of temporal and spatial artifacts, the framework enables explainable detection of AI-generated content.*

The **IVY-XDETECTOR** model follows a three-stage progressive training framework:
1.  **General Video Understanding:** Equipping the model with fundamental video comprehension.
2.  **AIGC Detection Fine-tuning:** Specializing the model for binary AIGC discrimination (real vs. fake).
3.  **Joint Optimization for Detection and Explainability:** Enabling the model to generate high-quality explanations while maintaining detection accuracy.

## Getting Started

### Dataset Access
The Ivy-Fake dataset can be accessed on Hugging Face:
[https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake](https://huggingface.co/datasets/AI-Safeguard/Ivy-Fake)

### Code
The code for the IVY-XDETECTOR and related experiments can be found in this repository:
[https://github.com/Pi3AI/IvyFake](https://github.com/Pi3AI/IvyFake) *(Further instructions on setting up the environment, running the code, and training/evaluating models will be provided here or in a separate `CONTRIBUTING.md` or documentation files.)*

## Citation

If you use Ivy-Fake or IVY-XDETECTOR in your research, please cite our paper:
```bibtex
@article{jiang2025ivyfake,
  title     = {Ivy-Fake: A Unified Explainable Framework and Benchmark for Image and Video AIGC Detection},
  author    = {Jiang, Changjiang and Zhang, Wayne and Zhang, Zhonghao and Yu, Fengchang and Peng, Wei},
  year      = {2025}
}
