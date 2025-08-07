# 🩺 Chest X-Ray Pneumonia Classifier using CNN

A real-time deep learning web app built with TensorFlow, Streamlit, and OpenCV that classifies chest X-ray images into **Normal** or **Pneumonia** using a Convolutional Neural Network (CNN).

---

## 🚀 Demo

![Demo GIF](demo.gif) <!-- Optional: Replace with your demo gif or remove -->

---

## 🧠 Model Summary

- **Architecture**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow 2.10
- **Dataset**: [Kaggle Chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Input Size**: 150x150 pixels
- **Classes**: `Normal`, `Pneumonia`

---

## 🏗️ Project Structure

```
medical-xray-cnn/
│
├── data/                         # Dataset and saved models
│   ├── eda.ipynb
│   └── saved_models/
│
├── src/
│   ├── train.py                 # Model training
│   ├── model.py                 # CNN architecture
│   ├── data_loader.py          # Data loading logic
│   └── predict.py              # Prediction logic
│
├── streamlit_app/
│   └── app.py                  # Streamlit frontend app
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/medical-xray-cnn.git
cd medical-xray-cnn
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For macOS/Linux
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (if not using a saved model)
```bash
cd src
python train.py
```

### 5. Run the Streamlit app
```bash
cd ..
streamlit run streamlit_app/app.py
```

---

## 📦 Dependencies

- tensorflow==2.10.0
- numpy==1.23.5
- opencv-python==4.6.0.66
- pandas==1.5.1
- scikit-learn==1.1.3
- matplotlib==3.6.2
- streamlit

---

## 📌 TODOs

- [ ] Enhance UI design (modern theme, color, layout)
- [ ] Add Grad-CAM or saliency map visualization
- [ ] Deploy on Hugging Face / Render / Streamlit Cloud
- [ ] Improve model accuracy with data augmentation

---

## 📸 Sample Predictions

| Chest X-ray | Prediction |
|-------------|------------|
| ![](sample1.png) | Pneumonia |
| ![](sample2.png) | Normal    |

---

## 🧑‍💻 Author

**Dhayanidhi P.**  
_MSc Data Science | Deep Learning & AI Enthusiast_  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
