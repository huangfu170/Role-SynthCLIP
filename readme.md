[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@yxhuangfu/Role-SynthCLIP/overview)

Official implementation of **Role-SynthCLIP**, a novel data synthesis framework that leverages multi-perspective role-playing prompts to generate semantically diverse captions for vision-language model (VLM) training.

**🌟 Key Highlights**
Semantic Diversity: Uses 5 expert roles to generate multi-perspective captions, solving the problem of shallow/repetitive descriptions in traditional synthetic data.
Data Efficiency: Achieves SOTA performance with only 1M synthetic pairs, outperforming baselines trained on 5M+ pairs.
Strong Performance: CLIP-B/16 trained on 1M Role-SynthCLIP pairs reaches 64.1% Recall@1 on MS COCO val, surpassing existing synthetic data baselines.
Robust Generalization: Excels on out-of-distribution (OOD) tasks, demonstrating enhanced cross-modal alignment and representation diversity.

**📖 Abstract**
Contrastive Language-Image Pretraining (CLIP) models rely heavily on the semantic diversity and quality of training data. Existing synthetic data methods focus on volume but lack diversity, leading to redundant captions. Role-SynthCLIP addresses this by guiding Multimodal Large Language Models (MLLMs) with role-playing prompts (e.g., Compositional Analyst, Narrative Setter) to generate fine-grained, multi-perspective image-text pairs. This approach improves caption expressiveness and alignment without increasing data volume, enabling efficient VLM training with limited resources.
**🚀 Quick Start**
**Installation**

```bash
git clone https://github.com/huangfu170/Role-SynthCLIP.git
cd Role-SynthCLIP
```

```bash
pip install -r requirements.txt
```

**Data Preparation**
Training Data: We use the ShareGPT4V dataset (1M images) for training. Download it from ShareGPT4V Official
Synthetic Caption Generation: Run the role-based caption generation pipeline
```bash
python scripts/generate_captions.py --image_dir path/to/sharegpt4v/images --output_dir data/synthetic_captions
```

Filtering: Apply Role-Aware Filter to clean noisy pairs
```bash
python scripts/role_aware_filter.py --input_dir data/synthetic_captions --output_dir data/filtered_pairs
```
Model Training
Train Role-SynthCLIP with CLIP-B/16 (default configuration)
```bash
python train.py \
  --data_path data/filtered_pairs \
  --model_type clip-b16 \
  --batch_size 256 \
  --epochs 6 \
  --lr 1e-6 \
  --output_dir checkpoints/role-synthclip-b16
```

**Evaluation**
Evaluate zero-shot retrieval on MS COCO
```bash
python evaluate.py \
  --model_path checkpoints/role-synthclip-b16 \
  --dataset coco \
  --split val \
  --metric recall@1
```

**📊 Experimental Results**
### Zero-shot Retrieval (Recall@1)

| Model | Data Size | MS COCO (I→T) | MS COCO (T→I) | Avg |
|-------|-----------|---------------|---------------|-----|
| CLIP-B/16 | 400M | 53.1 | 32.7 | 58.87 |
| FIX-CLIP | 5M | 61.3 | 47.0 | 75.95 |
| Role-SynthCLIP | 1M | 64.1 | 43.2 | 77.01 |

### Zero-shot Classification (Top-1 Accuracy)

| Model | ImageNet-1k | ImageNet-O | CIFAR-100 | Avg |
|-------|-------------|------------|-----------|-----|
| CLIP-B/16 | 68.3 | 40.4 | 66.7 | 70.30 |
| Role-SynthCLIP | 64.8 | 44.5 | 68.2 | 69.62 |

### 🔧 Core Components
1. Expert Roles
Role-SynthCLIP uses 5 complementary expert roles to generate diverse captions:
Observer of Details: Focuses on micro-level visual attributes (objects, colors, textures)
Interpreter of Context: Interprets situational meaning, human expressions, and cultural references
Compositional Analyst: Analyzes macro-level structure (spatial relationships, balance, perspective)
Narrative Setter: Synthesizes details into a coherent story or scene context
Emotional Responder: Captures mood, atmosphere, and aesthetic qualities
2. Framework Pipeline
Expert Role Generation: Define structured roles with specialized prompts
Multi-Perspective Captioning: Use Qwen2.5 VL to generate role-aligned captions
Role-Aware Filtering: Distill GPT-5's judgment to filter inaccurate/role-inconsistent pairs
CLIP Training: Extend positional embeddings for long captions and use multi-positive contrastive loss

### 📁 Project Structure
```plaintext
Role-SynthCLIP/
├── scripts/                # Utility scripts
│   ├── generate_captions.py  # Role-based caption generation
│   ├── role_aware_filter.py  # Filtering pipeline
│   └── data_preprocess.py    # Data formatting tools
├── models/                 # Model definitions
│   ├── clip_extended.py    # CLIP with long text support
│   └── role_aware_filter.py # Filter model
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── configs/                # Training configurations
```

### 🎯 Key Hyperparameters
Parameter	Value	Description
Global Batch Size	2048	Training batch size
Epochs	6	Number of training epochs
Learning Rate	1e-6	Initial learning rate
Weight Decay	1e-2	Weight decay for regularization
Max Sequence Len	248	Extended text sequence length

### 📚 Citation
If you use this work, please cite our paper:
bibtex
@article{huangfu2025rolesynthclip,
  title={Role-SynthCLIP: A Role Play Driven Diverse Synthetic Data Approach},
  author={Huangfu, Yuanxiang and Wang, Chaochao and Wang, Weilei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🤝 Contact
For questions or issues, please open an issue or contact:
Yuanxiang Huangfu: huangfuyuanxiang@patsnap.com
Chaochao Wang: wangchaochao@patsnap.com
Would you like me to add more details, such as a troubleshooting section, dataset download instructions, or visualization examples? I can also help refine the script commands or add a demo section for quick testing.
