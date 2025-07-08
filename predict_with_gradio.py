import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# Define your model
# -----------------------------
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Load the model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicCNN()
model.load_state_dict(torch.load("ai_vs_real_cnn.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class_names = ['AI-Generated Art', 'Human-Made Art']

# -----------------------------
# Prediction function
# -----------------------------
def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        pred_label = class_names[pred_idx]
        confidence = probs[pred_idx].item()
        return f"🖼️ {pred_label} (Confidence: {confidence:.2%})"

# -----------------------------
# Load example images if present
# -----------------------------
examples = None
if os.path.isdir("examples"):
    examples = [["examples/" + f] for f in os.listdir("examples") if f.endswith((".jpg", ".png"))]

# -----------------------------
# Clean CSS with correct selectors
# -----------------------------
css = """

/* === Only style the app, NOT Gradio system panels === */
.gradio-container {
    background-color: black !important;
    color: white !important;
    padding: 20px;
    border-radius: 12px;
}

/* === Headings & Paragraphs in the app === */
.gradio-container h1,
.gradio-container h2,
.gradio-container p {
    color: white !important;
    text-shadow: 1px 1px 2px black;
}

/* === 'by Bareera Mushthak' and Markdown blocks === */
.gradio-container .markdown {
    color: white !important;
}

/* === Upload box and label === */
#upload-box {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    padding: 10px;
    border: none !important;          /* Remove harsh border */
    box-shadow: 0 0 10px rgba(72, 209, 204, 0.4);  /* Soft glow */
}
#upload-box label {
    background-color: white !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 2px;
}

/* === Prediction output box === */
#output-box {
    background-color: white !important;
    border-radius: 6px;
    padding: 10px;
    border: none !important;
}

#output-box textarea {
    background-color:#1E6981  !important;
    color: white !important;
    font-weight: bold;
    font-size: 20px;
    border:3px white !important;
    outline: none !important;
    resize: none;
}
label[for="output-box"] {
    color: black !important;
    font-weight: bold !important;
    font-size: 20px !important;
}
/* === Classify Button === */
#classify-button {
    background-color: orange !important;
    color: black !important;
    font-weight: bold;
    border-radius: 6px;
    border: none;
}

/* === DO NOT TOUCH SETTINGS PANEL === */
"""


# -----------------------------
# Gradio interface
# -----------------------------
with gr.Blocks(css=css) as demo:
    gr.Markdown("""
## 🎨 AI vs Human Art Classifier  
<span style='font-size:16px; color:white;'>by Bareera Mushthak</span>
""")

    gr.Markdown("Upload a painting and click Classify to find out whether it's AI-generated or human-made.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Artwork",elem_id="upload-box")
        output_box = gr.Textbox(label="Prediction", lines=2, interactive=False, elem_id="output-box")

    button = gr.Button("Classify", elem_id="classify-button")
    button.click(fn=predict, inputs=image_input, outputs=output_box)

    if examples:
        gr.Examples(examples, inputs=image_input)
    # to give footer
    gr.Markdown("<p style='text-align:center; font-size:14px; color:#cccccc;'>© 2025 Bareera Mushthak | AI Developer</p>")


# -----------------------------
# Launch with public link
# -----------------------------
demo.launch(share=True)
