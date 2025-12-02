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
```

Note: You may need to restart your kernel after installing the dependencies.

## 🏁 Usage

1. Clone the Repository:

```bash
git clone https://github.com/krish-25k/Image-Captioning-Model-.git
cd Image-Captioning-Model-
```

2. Open the Notebook:
Launch Jupyter Notebook or Jupyter Lab and open Image_Captioning_Transformer.ipynb.

3. Run the Cells:
Execute the cells in order. The notebook will automatically download the necessary weights for ResNet, YOLOv8, and BLIP.

4. Test with Your Images:
Scroll to the bottom of the notebook to find the Interactive UI. Upload an image to see:
   * The raw generated caption.
   * The "humanized" version (via GPT-2).
   * A list of objects detected by YOLOv8.

## 📊 Dataset

This project uses the Flickr8k dataset, which consists of 8,000 images each paired with 5 different captions. It serves as the standard benchmark for training image captioning models.

## 🤖 Future Improvements

* Implement METEOR and BLEU score evaluation metrics.
* Deploy the model as a web app using Streamlit or Gradio.
* Fine-tune the BLIP model on specific domains (e.g., medical imaging).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.