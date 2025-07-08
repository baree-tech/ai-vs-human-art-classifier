# 🖼️ AI vs Human Art Classifier 🎨
A deep learning model to classify paintings as **AI-generated** or **human-created** — built using **PyTorch**, trained on a custom dataset, and deployed with **Gradio** for real-time testing.

---

## 🔍 Motivation

With the rise of generative AI tools like DALL·E, Midjourney, and Stable Diffusion, distinguishing human-made art from AI-generated imagery is becoming increasingly important for **art authenticity**, **ethics**, and **copyright integrity**.

This project aims to tackle this by building a **binary classifier** that predicts whether a painting is AI-generated or created by a human artist.

---

## 📦 Dataset

A fully **custom-curated dataset** built by combining:

### 🧠 Human Art:
- Sourced from **WikiArt (Kaggle)**
- 20+ styles: Abstract Expressionism, Cubism, Nouveau, etc.
- Manually organized into `/human/<style_name>` subfolders

### 🤖 AI Art:
- Sourced from **AI Art Archives / Diffusion Repositories**
- 25+ styles including Stable Diffusion outputs
- Folders: `/AI/ai-diffusion-db-large1`, `/AI/...`

#### ✅ Preprocessing:
- Only selected folders with ≥500 images/class
- Created stratified split for train (70%), val (20%), test (10%)
- Final size: **~15,000 images**
- Applied normalization and data augmentation (`transforms`)

---

## 🧠 Model: Basic CNN

```python
3 Convolutional Layers
+ ReLU Activation
+ MaxPooling
+ Dropout (0.5)
+ Fully Connected Dense Layer
```

Trained from scratch on GPU using **CrossEntropyLoss + Adam Optimizer**

---

## 📈 Results

| Metric           | Score     |
|------------------|-----------|
| **Train Accuracy** | ~96.76%      |
| **Validation Accuracy** | ~94.15%   |
| **Test Accuracy** | ~94.2%    |

Also includes:
- 🔍 **Classification Report**
- 📉 **Confusion Matrix**
- 🖼️ **Sample Predictions Visualization**

---

## 🧪 Personal Validation

🎨 I tested the model with my **own hand-painted landscape artworks**, and it **successfully classified them as human-made** via the Gradio app.

---

## 🚀 Gradio Deployment

The project is deployed via **Gradio**, which allows you to:
- Upload any painting or image
- Instantly get prediction (AI vs Human)
- Try with your own artwork!

```bash
gr.Interface(fn=predict_image, inputs="image", outputs="text").launch(share=True)
```

> 🟢 App will run locally and optionally generate a public link.

---

## 💾 Model Saving & Loading

```python
# Save
torch.save(model.state_dict(), 'ai_vs_human_cnn.pth')

# Load for prediction
model.load_state_dict(torch.load('ai_vs_human_cnn.pth'))
```

---

## 📁 Project Structure

```
AI_vs_Real/
│
├── dataset_split/
│   ├── train/AI/
│   ├── train/human/
│   ├── val/...
│   └── test/...
│
├── main.py                    # Model training & saving
├── evaluate_model.py          # Confusion matrix, classification report
├── predict_with_gradio.py     # Gradio app for live prediction
└── README.md
```

---

## 🧠 Future Improvements

- Try **transfer learning (ResNet50, EfficientNet)**
- Extend to **multi-class: specific styles (e.g., Cubism vs Impressionism)**
- Add **explainability** (e.g., Grad-CAM)

---

## 👩‍🎨 Created by: Bareera Mushthak

> Follow me on [LinkedIn](https://www.linkedin.com/in/bareera-mushthak)  
> Let’s build responsible AI together 🧠✨

---

## 🏷️ Tags

`#PyTorch` `#ComputerVision` `#AIArt` `#DeepLearning` `#Gradio` `#EthicalAI` `#ImageClassification`
