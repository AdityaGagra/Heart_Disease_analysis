# Heart_Disease_analysis
Heart Disease Prediction App 🫀
This repository contains a Machine Learning application built with Streamlit that predicts the likelihood of heart disease in patients based on medical attributes. The model is trained using a Random Forest classifier (as seen in the provided Jupyter Notebook).

🚀 Project Overview
The goal of this project is to provide an easy-to-use interface where users can input clinical data (such as age, cholesterol, and maximum heart rate) to get an instant prediction.

📂 File Structure
app.py: The main Streamlit web application.

Heart_Disease.ipynb: The Jupyter Notebook containing data exploration, visualization, and model training.

heart.csv: The dataset used for training and testing the model.

heart_disease_model.pkl: The serialized (saved) machine learning model.

requirements.txt: A list of Python libraries needed to run the app.

🛠️ Installation & Setup
To run this project locally, follow these steps:

1. Clone the repository:
git clone https://github.com/AdityaGagra/Heart_Disease_analysis.git
cd Heart_Disease_analysis

3. Install dependencies:
pip install -r requirements.txt

4.Run the application:
streamlit run app.py


📊 Dataset Features
The model uses several clinical features, including:

Age: Age of the patient.

Sex: Gender (1 = male; 0 = female).

CP: Chest pain type.

Trestbps: Resting blood pressure.

Chol: Serum cholesterol in mg/dl.

Thalach: Maximum heart rate achieved.

...and more.

🧠 Model Training
The model was trained using the Random Forest algorithm, achieving high accuracy by analyzing patterns in the heart.csv dataset. Detailed evaluation metrics and visualizations can be found in Heart_Disease.ipynb.
