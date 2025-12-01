# 🖼️ Image Captioning with PyTorch & Transformers

## 📌 Project Overview
This project demonstrates end-to-end **image captioning** using both a custom **CNN-RNN model with Attention** and state-of-the-art **Transformer models (BLIP)**. It also integrates **YOLO object detection** to enrich captions and includes **caption humanization** for more natural language output.

The complete implementation is provided in a Google Colab notebook, making it easy to explore, train, and experiment with different models.

---

## 🚀 Key Features

### 🧠 Deep Learning Captioning Models
- **CNN-RNN Encoder-Decoder with Attention**
  - Flexible CNN Encoders: *ResNet*, *EfficientNet*, *ConvNeXt*
  - RNN Decoder: *LSTM* or *GRU*
  - Bahdanau-style additive attention

- **Transformer Model (BLIP)**
  - Integrates `Salesforce/blip-image-captioning-base` via Hugging Face
  - High-quality captions out-of-the-box

### 👁️ Object-Aware Captioning
- **YOLOv8 (yolov8n.pt)** detects prominent image objects
- Detected objects help refine captions and provide suggestion prompts

### 🧩 Additional Enhancements
- Caption humanization with rule-based text polishing
- Smart word suggestions based on detected objects + caption context
- Custom vocabulary management for CNN-RNN model tokenization

### 📚 Dataset & Evaluation
- Training supported with **Flickr8k** dataset (`datasets` library)
- Includes **BLEU score** and **precision** evaluation metrics

---

## 🛠️ Setup & Installation

1. **Open in Google Colab**
   - Upload the notebook or click the “Open in Colab” badge (if included)

2. **Install Dependencies**
   Run the first code cell to install:
   - `torch`, `torchvision`, `transformers`, `ultralytics`
   - `datasets`, `nltk`, `ipywidgets`, and more

3. **Restart Kernel** *(Important!)*

Required to correctly load newly installed packages

4. **NLTK Setup**
- `punkt` tokenizer is downloaded automatically during execution

---

## ▶️ Usage Instructions

1. **Run All Cells**
This will:
- Load and preprocess dataset
- Initialize `Vocabulary`, CNN-RNN and BLIP models
- Train models (optional — BLIP works pre-trained)
- Activate the image upload widget

2. **Generate Captions Interactively**
- Upload an image using the UI widget
- The notebook will:
  - Display the image
  - Generate captions from both BLIP and CNN-RNN
  - Show humanized caption variants
  - Display YOLO-detected objects
  - Provide smart suggestions for improving captions

3. **Test Humanization Module**
You may call:
```python
test_humanize_captions()
