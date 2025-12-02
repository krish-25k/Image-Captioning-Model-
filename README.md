# 🖼️ Smart Image Captioning & Object Detection

A hybrid deep learning project that combines **Computer Vision** and **Natural Language Processing** to generate descriptive captions for images. This repository implements multiple architectures—from custom CNN-RNN models with Attention to state-of-the-art Transformers—and integrates Object Detection for enhanced context.

## 🚀 Key Features

* **Multi-Model Approach**:
    * **Custom CNN-RNN**: Uses **ResNet50** (encoder) and **LSTM/GRU with Attention** (decoder) to generate captions from scratch.
    * **Transformer Power**: Integrates **BLIP** (Bootstrapping Language-Image Pre-training) for high-fidelity captioning.
* **Object Detection Integration**: Utilizes **YOLOv8** to detect and list objects within the image, providing granular scene understanding.
* **Human-Like Text**: Includes a post-processing pipeline using **GPT-2** to "humanize" and refine generated captions for better grammar and flow.
* **Interactive UI**: Features a built-in GUI (using `ipywidgets`) allowing users to upload images, view predictions, and manually edit captions with smart suggestions.

## 🛠️ Tech Stack

* **Deep Learning Framework**: PyTorch
* **Encoders**: ResNet50, EfficientNet, ConvNeXt
* **Decoders**: LSTM, GRU, Transformer (BLIP)
* **Object Detection**: Ultralytics YOLOv8
* **NLP**: Hugging Face Transformers (GPT-2, BERT tokenizer)
* **Dataset**: Flickr8k
* **Tools**: NumPy, Pandas, Matplotlib, PIL

## 📂 Project Structure

The notebook is organized into the following modules:

1.  **Data Loading & Preprocessing**:
    * Loads the Flickr8k dataset.
    * Builds a vocabulary from training captions.
    * Applies image transforms (Resize, Crop, Normalize).
2.  **Model Architectures**:
    * **Encoder**: Extracts feature vectors from images.
    * **Attention Mechanism**: Focuses on specific image regions during text generation.
    * **Decoder**: Generates caption sequences word-by-word.
3.  **Training Loop**:
    * Implements CrossEntropyLoss and AdamW optimizer.
    * Includes validation phases to monitor performance.
4.  **Inference & UI**:
    * Beam Search implementation for better caption generation.
    * Interactive widget for real-time testing.

## 📦 Installation

To run this notebook locally or on Colab, ensure you have the required dependencies:

```bash
pip install torch torchvision transformers ultralytics datasets ipywidgets nltk matplotlib

