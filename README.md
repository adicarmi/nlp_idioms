# Picture This: Disambiguating Literal and Figurative Idioms with Text and Images  

[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Kd3BURLIZVA6dzKSVYLairIWo_KLMZfh?usp=sharing)  

Link to colab: https://colab.research.google.com/drive/1Kd3BURLIZVA6dzKSVYLairIWo_KLMZfh?usp=sharing

This repository contains the code, datasets, and experiments for our project:  
**_Picture This: Disambiguating Literal and Figurative Idioms with Text and Images_**.  

We propose a multimodal approach to distinguish literal from figurative idioms using text and images. By fine-tuning **MobileCLIP**, a lightweight CLIP variant, we align idiomatic text spans with both **literal** and **figurative** image anchors. Our results demonstrate significant improvements in idiom disambiguation through visual grounding.

---

## 🚀 Motivation

Idioms such as _“kick the bucket”_, _“blind as a bat”_, or _“raise the roof”_ are often **non-compositional**—their meanings cannot be derived from individual words alone. Even large language models often fail to distinguish literal vs. figurative uses.  

Example:  
- Literal: _“He kicked the bucket in the yard.”_  
- Figurative: _“He finally kicked the bucket.”_  

To tackle this, we leverage images as **semantic anchors**—echoing the phrase: _“a picture is worth a thousand words.”_  

---

## 📚 Datasets

- **Custom Idiom Images Dataset**  
  - 36 idioms sampled from the **MAGPIE corpus**.  
  - For each idiom: 2 generated images (literal + figurative) created using GPT-4.1 prompts + Hugging Face FLUX.  

- **MAGPIE Corpus** ([Haagsma et al. 2020])  
  - Gold standard dataset for idiom disambiguation.  
  - Span-level annotations for literal vs. figurative usages.  
  - 1,180 labeled instances curated for this project.  

- **MS COCO Captions**  
  - Subset of 1,000 image-caption pairs.  
  - Used for auxiliary **image–text retrieval** objective.  

---

## 🏗️ Methodology

We fine-tune **MobileCLIP (S0)** with:  
- **Span embeddings**: token-level idiom representations from text.  
- **Image embeddings**: fixed literal/figurative anchors from MobileCLIP’s image encoder.  

### Training Losses
1. **Span-to-image cross-entropy (InfoNCE)** → Align idiom spans with literal/figurative anchors.  
2. **Supervised Contrastive Loss** → Tight intra-idiom clusters, broad inter-idiom grouping.  
3. **KL-Divergence Regularization** → Prevent catastrophic forgetting.  
4. **Pairwise Margin Loss** → Enforce separation between correct and incorrect anchors.  
5. **Auxiliary COCO Retrieval Loss** → Maintain multimodal alignment.  

**Total loss** = Weighted combination of all above.  

---

## ⚙️ Implementation

- **Frameworks**: PyTorch Lightning, Hugging Face Transformers, Pandas, scikit-learn, matplotlib.  
- **Training Setup**:  
  - Environment: Google Colab, NVIDIA T4 GPU (16GB).  
  - Optimizer: AdamW, cosine LR schedule.  
  - Precision: Mixed (16-bit).  
  - Batch size: 16, epochs: 10.  
  - Training time: ~7 min per model configuration.  

---

## 📊 Results

### Idiom Disambiguation
| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| **Baseline (pretrained CLIP)** | 0.51 | 0.52 |
| **Fine-tuned (Pair Margin)** | **0.90** | **0.90** |
| Full Model (all losses) | 0.89 | 0.89 |

### COCO Retrieval
- Baseline performed best on Recall@5 (~0.77).  
- Fine-tuned models slightly lower (~0.61–0.76). Auxiliary COCO loss helped to preserve performance.  

---

## 🔎 Error Analysis

- **Common issue**: Literal usages often misclassified as figurative (_e.g., “add fuel to the fire” used literally_).  
- **Ambiguous idioms** (like _“turn the corner”_ or _“rock the boat”_) remain challenging.  
- Dataset imbalance (more figurative samples) inflated metrics for single-sense idioms.  

Despite these, fine-tuning corrected systematic baseline errors and improved per-idiom F1 for many cases (e.g., _“break the bank”_, _“in a rut”_).

---

## 🧾 Key Findings

- Visual grounding **substantially improves idiom disambiguation**.  
- Pairwise margin + supervised contrastive losses drive the biggest gains.  
- Auxiliary COCO retrieval stabilizes multimodal alignment.  
- Literal senses remain harder → need better balancing and augmentation.  

---

## 🔮 Future Work

- Expand to **larger and more balanced idiom corpora**.  
- Dynamic loss weighting (e.g., Optuna).  
- Improved literal recognition with targeted augmentation.  
- Examine alternative losses: triplet, angular margin, center loss.   
- Extend framework to **metaphors, sarcasm, and figurative language beyond idioms**.  

---

## 🖇️ Citation

If you use this work, please cite:

```
@article{carmi2025picture,
  title={Picture This: Disambiguating Literal and Figurative Idioms with Text and Images},
  author={Carmi, Adi and Hanani, Omer},
  year={2025}
}
```

---

## 🔗 Links

- ▶️ [Run the Colab Notebook](https://colab.research.google.com/drive/1Kd3BURLIZVA6dzKSVYLairIWo_KLMZfh?usp=sharing)  
