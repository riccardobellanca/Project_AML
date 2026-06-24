# Parameter-Efficient Adaptation of DINOv2 for Sub-Pixel Semantic Correspondence

> **Advanced Machine Learning — A.Y. 2025/2026**  
> Politecnico di Torino

---

## 👥 Team

| Johnprice Osagie | Mario Lapadula | Giorgia Pugliese | Riccardo Bellanca |

---

## 🎯 Task Description

This paper presents a modular pipeline optimizing **DINOv2** for sub-pixel semantic correspondence through **LoRA** and **Curriculum Learning**. To resolve background clutter and quantization errors, it pairs **SAM**-based masking with a novel entropy-driven **Adaptive Window Soft-Argmax** for superior precision.

---

## 🗂️ Repository Structure

```
Project_AML/
├── dataloaders/                 # Data loading scripts (SPair-71k, PF-Pascal)
├── models/                      # DINOv2 Extractor, LoRA & Correspondence models
├── utils/                       # Metrics, Soft-Argmax, Curriculum & SAM masking
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
└── README.md
```

---

## 🧠 Approach

### 1. Backbone & PEFT Adaptation (LoRA & BitFit)
We adopt a pre-trained **DINOv2 (ViT-B/14)** backbone. To specialize the representations without full fine-tuning, we explore:
- **Low-Rank Adaptation (LoRA)**: Injects trainable low-rank matrices into attention projections ($\Delta W = BA$ with rank $r=16$), optimizing only $1.03\%$ of the backbone.
- **BitFit**: Updates solely the network’s bias parameters, optimizing a mere $0.09\%$ of parameters.

### 2. Curriculum Learning
To prevent gradient instability, we dynamically rank training pairs by scale, truncation, and viewpoint difficulty using SPair-71k metadata. A linear pacing function progressively introduces harder transformations over the first 10 epochs.

### 3. Inference Refinement Pipeline
- **Global Segment-Aware Masking (SAM)**: Eliminates background false positives by setting similarity scores outside SAM-generated target masks to $-\infty$.
- **Local Adaptive Window Soft-Argmax**: Mitigates coarse-grid quantization errors. It dynamically scales the soft-argmax search window based on the Shannon entropy of the match probability, achieving differentiable, continuous sub-pixel accuracy.

---

## 📐 Evaluation — PCK

We measure matching accuracy using the **Percentage of Correct Keypoints (PCK)**, evaluating at both standard ($\alpha=0.1$) and strict localization ($\alpha=0.05$) thresholds across:
- **SPair-71k**: Primary training and evaluation benchmark.
- **PF-Pascal**: Used to verify zero-shot cross-dataset generalization.

---

## 📚 References

1. Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, and Nicolas Ballas. *Self-supervised learning from images with a joint-embedding predictive architecture.* In CVPR, 2023.
2. Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. *Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models.* 2022.
3. Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. *Curriculum learning.* In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.
4. Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Dídac Surís, Chaitanya K Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. *Sam 3: Segment anything with concepts.* arXiv preprint arXiv:2511.16719, 2025.
5. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. *Masked autoencoders are scalable vision learners.* In CVPR, 2022.
6. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. *Lora: Low-rank adaptation of large language models.* In ICLR, 2022.
7. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. *Segment anything.* In ICCV, 2023.
8. Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. *Spair-71k: A large-scale benchmark for semantic correspondence.* In ICCV, 2019.
9. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Hervé Soyer, Vighnesh Yashar, et al. *Dinov2: Learning robust visual features without supervision.* arXiv preprint arXiv:2304.07193, 2023.
10. Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. *Emergent correspondence from image diffusion.* In NeurIPS, 2023.
11. Junyi Zhang, Charles Herrmann, Junhwa Jun, Richard Wood, Koray Kavukcuoglu, et al. *A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence.* In NeurIPS, 2023.
12. Junyi Zhang, Charles Herrmann, Junhwa Jun, Richard Wood, and Koray Kavukcuoglu. *Telling left from right: Identifying geometry-aware semantic correspondence.* In CVPR, 2024.
