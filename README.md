## Wine Quality Prediction

This project predicts the quality of wine using both **Machine Learning (Random Forest)** and a **Deep Learning Neural Network**. It uses the **Wine Dataset** from `sklearn.datasets` and a **Streamlit web app** for user interaction.

---

### **Project Overview**

* Load and explore the **Wine Dataset**.
* Perform preprocessing (scaling and encoding).
* Train a **Random Forest Classifier** and a **Deep Learning model** using Keras.
* Build a **Streamlit app** where users can input wine features and get a predicted quality score.

---

### **Technologies Used**

* **Python**
* **Pandas**, **NumPy** (Data Handling)
* **Matplotlib**, **Seaborn** (Visualization)
* **scikit-learn** (Data preprocessing & Random Forest Classifier)
* **TensorFlow/Keras** (Neural Network model)
* **Streamlit** (Web app)
* **Joblib** (Model saving & loading)

---

### **Dataset**

The dataset comes from `sklearn.datasets.load_wine()`.

**Features (13):**

* alcohol
* malic\_acid
* ash
* alcalinity\_of\_ash
* magnesium
* total\_phenols
* flavanoids
* nonflavanoid\_phenols
* proanthocyanins
* color\_intensity
* hue
* od280/od315\_of\_diluted\_wines
* proline

**Target:** Wine quality class (0, 1, 2).

---

### **Model Training**

#### Random Forest Classifier:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

#### Deep Learning Model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(3, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

---

### **Saving Models**

```python
import joblib

#Save scaler and RandomForest
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(rf_model, 'rf_model.pkl')

#Save Keras model
model.save('wine_model.keras')
```

---

### **Streamlit Web App**

Users can input 13 wine features and predict wine quality:

```python
st.number_input("alcohol", min_value=0.0, step=0.1)
...
```

Prediction:

```python
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
st.write(f'Predicted Wine Quality: {np.argmax(prediction) + 1}')
```

---

### **Running the App**

1. Clone the repository:

   ```bash
   git clone https://github.com/BenedictOuma/Wine-Quality-Prediction.git
   cd wine_ANN
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run app.py
   ```

---

### **Example Outputs**

* Confusion Matrix and Classification Report for Random Forest
* Neural Network validation accuracy and loss plots

---

### **App Preview**

![app-preview](https://via.placeholder.com/800x400?text=Streamlit+Wine+App+Preview)

---

### **Contributing**

* Fork the repo
* Create a branch: `git checkout -b feature-name`
* Submit a Pull Request