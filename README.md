<div align="center">
    <h1>
    AMP: Wavelet-enhanced Foundation Model Adaptation for Weakly Supervised Urban Tree Canopy Mapping
    </h1>
    <h3><b>Accepted by IEEE Transactions on Geoscience and Remote Sensing 2026</b></h3>
    <div>
        <a href='https://github.com/Tengmingn' target='_blank'>Mingnuo Teng<sup>1*</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=xY0__WQAAAAJ&hl=zh-CN&oi=sra' target='_blank'>Jianhua Guo<sup>2*</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=1umAObUAAAAJ&hl=zh-CN&oi=sra' target='_blank'>Huanjing Yue<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=x0jlbE4AAAAJ&hl=zh-CN&oi=sra' target='_blank'>Jingyu Yang<sup>1&#8224</sup></a>
    </div>
    <div>
        <sup>1</sup>Tianjin University,&emsp;<sup>2</sup>Aerospace Information Research Institute
    </div>
    <div>
        <h4 align="center">
            <!-- <a href="https://github.com/Tengmingn/AMP" target='_blank'>
                <img src="https://img.shields.io/badge/🌟-Project%20Page-blue" style="padding-right: 20px;">
            </a> -->
            <a href="https://doi.org/10.1109/TGRS.2026.3693826" target='_blank'>
                <img src="https://img.shields.io/badge/IEEE_TGRS-2026-b31b1b.svg" style="padding-right: 20px;">
            </a>
        </h4>
    </div>
</div>


## 🌟 Method Overview
This repository contains the official PyTorch implementation of **AMP**, a framework designed for high-accuracy urban tree canopy mapping using weakly supervised sparse annotations. By integrating wavelet-enhanced adaptation with foundation models (DINOv2), our method effectively captures fine-grained spatial details in complex urban environments.

![Method Overview](./assets/Method.png)

---
## 📷 Result Display
![Method Overview](./assets/comparison_UTC.png)
---

## 🛠️ Installation & Training & Inference

### 1. Clone the repository
```bash
git clone https://github.com/Tengmingn/AMP.git
cd AMP
conda create -n amp python=3.10
conda activate amp
pip install -r requirement
```
### 2. Download the UTC-Sparse dataset
Get our dataset at https://ieee-dataport.org/documents/utc-sparse
### 3. Prepard third party code
```
cd third_party
# dinov2
git clone https://github.com/facebookresearch/dinov2.git
# dinov3 if you want
git clone https://github.com/facebookresearch/dinov3.git
```
Then, insert the `user_forward_features` function into the respective model files. The corresponding function codes are provided in `./third_party/dinov2_userfunc` and `./third_party/dinov3_userfunc`.

Alternatively, you can download our pre-packaged third-party folder and extract it directly into the `./third_party` directory.
`https://pan.baidu.com/s/1Y_2E5OTQpKRoBlxD-HDMdw?pwd=57kr` password: 57kr
### 4. Reproduce
Our pre-trained and fine-tuned models are available at: `https://pan.baidu.com/s/1TzAoo50w1w1lWD5H_yIrHA?pwd=edhq` password: edhq
### 5. Training on your own dataset
### 6. Inference
## ❤️ Acknowledgement

We sincerely acknowledge the following works for their valuable contributions and inspiration to this project:

- **SparseFormer: A Credible Dual-CNN Expert-Guided Transformer for Remote Sensing Image Segmentation With Sparse Point Annotation (TGRS 2025)** — for providing the implementation of the multi-branch pseudo-label generation strategy.(https://github.com/Yujia73/SparseFormer)

- **Sparsely Annotated Semantic Segmentation with Adaptive Gaussian Mixtures** — for providing the foundational network structure implementation.(https://github.com/Luffy03/AGMM-SASS)

- **DINOv2: Learning Robust Visual Features without Supervision** — for providing the pretrained visual feature backbone used in this work.(https://github.com/facebookresearch/dinov2)

We deeply appreciate the authors of these works for making their research and code publicly available to the community.

## 🎓 Citations

## 📧 Contact
If you have any inquiries, please reach out us via email at `tmn@tju.edu.cn`

