# 🎨 AI vs Human Art Classifier

A fully functional Gradio web app that classifies whether a painting is **AI-generated** or **human-made**, built with **PyTorch**, styled with custom CSS, and ready to deploy.

---

## 🧠 Motivation

With the rise of AI-generated art (DALL·E, Midjourney, Stable Diffusion), distinguishing AI from human art is harder than ever. This project:
- Trains a CNN model to classify artwork origin.
- Builds a minimal but stylish interface.
- Demonstrates end-to-end AI deployment skills.

---

## 🏗️ Project Overview


### 🖼️ Dataset

The original dataset was approximately **67 GB** in size, containing thousands of AI-generated and human-created artworks, each organized into 25 style-based folders under `AI/` and `human/` categories.

#### Why We Subsampled:

Due to the large size and volume of images, **training directly on the full dataset led to system instability and memory issues**. To make the training process manageable and efficient, we:

- **Selected 500 images per class**, for both AI and Human art categories.
- This resulted in:
  - **12,500 AI-generated images** (25 classes × 500)
  - **12,500 Human-created images** (25 classes × 500)

#### Structure:

- `AI/`
  - 25 style folders (e.g., *ai-abstract-expressionism, ai-cubism, ai-surrealism*, etc.)
- `human/`
  - 25 style folders (e.g., *human-impressionism, human-realism*, etc.)

#### Dataset Split:

To ensure proper generalization and evaluation:
- **Training Set**: 70%
- **Validation Set**: 20%
- **Test Set**: 10%

This balanced and preprocessed dataset improved classification performance and reduced computational overhead.


### 🧠 Model (BasicCNN)
- 3 convolutional layers + pooling
- Fully connected layers with dropout
- Final output: 2 classes (AI-generated or Human-made)
- Achieved ~95% accuracy on validation set.

### 🎮 App Interface
- Built with **Gradio (Blocks API)**
- Interactive UI:
  - Upload a painting
  - Click **Classify**
  - View prediction with confidence score
- Styled with custom CSS:
  - Black dark theme background
  - Orange “Classify” button
  - Teal/white prediction box with bold text

---

## 🔧 How to Run Locally

```bash
git clone https://github.com/baree-tech/ai-vs-human-art-classifier.git
cd ai-vs-human-art-classifier
pip install -r requirements.txt
python predict_with_gradio.py
```
- Then go to `http://localhost:7860`

---

## 🚀 Deployment with Hugging Face Spaces

(You can deploy this app on Hugging Face Spaces using the Gradio interface)
- Use **Gradio SDK**
- Host publicly
- Provide `predict_with_gradio.py`, the `.pth` model, and `requirements.txt`

---

## 🧩 File Structure

```
AI-vs-human-art-classifier/
├── ai_vs_real_cnn.pth        ← Trained model weights
├── predict_with_gradio.py    ← Main app script
├── requirements.txt          ← Dependency list
├── README.md                 ← This documentation
└── examples/                 ← Optional: sample test images
```

---

## 🔍 Sample Output

```
🖼️ Human-Made Art (Confidence: 92.35%)
```

---

## About the Creator

Developed by **Bareera Mushthak**, an AI developer passionate about combining art and technology.  
Connect with me on [LinkedIn](https://www.linkedin.com/in/bareera-mushthak) | Explore more projects on [GitHub](https://github.com/baree-tech)

---

## 📝 License

MIT License © 2025 Bareera Mushthak
