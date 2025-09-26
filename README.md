## CNN Image Classification with MobileNet

This project implements an image classification model using TensorFlow/Keras, trained on the CIFAR dataset, and deployed with Streamlit for real-time predictions. The model is based on MobileNet, a lightweight and efficient CNN architecture.

## 🚀 Features

- Uses CIFAR dataset for training and validation

- MobileNet architecture for efficient deep learning

- Image preprocessing: resizing, normalization, and augmentation

- Training with Adam optimizer, categorical crossentropy, dropout, and early stopping

- Saved as a .keras model for reuse

- Deployed with Streamlit for interactive, real-time predictions

## 🚀 Getting Started

Follow these steps to set up and run the CNN image classification locally.

#1️⃣ Clone the Repository
```bash
git clone https://github.com/kirti111agarwal/CNN_image_classification.git
```

#2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#3️⃣ Run the App
```bash
streamlit run app.py
```
### 📂 Project Structure

├── preprocessing.ipynb # notebook for preprocessing
├── cnn_model.keras     # Saved .keras model  
├── app.py              # Streamlit deployment file  
├── requirements.txt    # Dependencies  
└── README.md           # Project documentation  

📊 Results

- Stable validation accuracy with reduced overfitting

- Improved generalization with dropout and early stopping

- Real-time predictions deployed with Streamlit
