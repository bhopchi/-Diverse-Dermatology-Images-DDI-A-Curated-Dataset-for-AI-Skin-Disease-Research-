# -Diverse-Dermatology-Images-DDI-A-Curated-Dataset-for-AI-Skin-Disease-Research-
AI may aid in triaging skin diseases, but biases exist due to limited diverse datasets. The Diverse Dermatology Images (DDI) dataset, with 656 pathologically confirmed images (570 patients) from Stanford Clinics (2010–2020), includes Fitzpatrick skin types (FST) I-VI. FST I-II and V-VI images were matched by diagnosis, age, gender, and date.


Here’s a draft README for the provided repository:

---

# Diverse Dermatology Images (DDI) Dataset and Model

This repository provides code and resources for working with the Diverse Dermatology Images (DDI) dataset, a curated and pathologically confirmed collection of skin disease images designed to address biases in AI algorithms for dermatological diagnosis. It includes tools for dataset preparation, training a MobileNet-based model, and evaluating performance.

---

## Dataset Overview

The DDI dataset contains:
- **656 images** representing **570 unique patients** from **Stanford Clinics (2010–2020)**.
- Images covering **Fitzpatrick skin types (FST) I-VI**.
- Balanced sampling between FST I-II and FST V-VI by matching diagnosis, age, gender, and photograph date.

### Class Distribution:
| Skin Type | Benign | Malignant |
|-----------|--------|-----------|
| FST I-II  | 159    | 49        |
| FST III-IV| 167    | 74        |
| FST V-VI  | 159    | 48        |

---

## Features

- **Data Preparation**: Scripts for unzipping and preprocessing the DDI dataset.
- **Model Training**: Uses **MobileNet** architecture with transfer learning.
- **Evaluation**: Includes metrics for accuracy and loss visualization.
- **Custom Prediction**: Tools for classifying new images using the trained model.
- **Callbacks**: Early stopping and model checkpointing for improved training.

---

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Dataset
Unzip the dataset and preprocess:
```python
unzip_dataset('ddidiversedermatologyimages.zip', 'dataset.DDI')
```

### 2. Load Data
Load and preprocess metadata:
```python
train_gen, valid_gen = load_data('dataset.DDI')
```

### 3. Train Model
Train the MobileNet-based model:
```python
model.fit(train_gen, validation_data=valid_gen, epochs=5)
```

### 4. Evaluate Model
Evaluate accuracy and loss:
```python
val_loss, val_accuracy = model.evaluate(valid_gen)
```

### 5. Make Predictions
Predict class for a new image:
```python
predicted_class = predict_image('path_to_image.png')
```

---

## Results

- **Training Samples**: 80% of the dataset
- **Validation Samples**: 20% of the dataset
- **Model Accuracy**: Shown in training plots.

---

## Contributions

Contributions are welcome! If you find any issues or have suggestions, please open an issue or a pull request.

---

## License

This repository is open-sourced for academic and research purposes. Ensure ethical use when working with the dataset and code.

---

Feel free to adjust details specific to your project or dataset!
