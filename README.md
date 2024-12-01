# Fish Image Classification with Artificial Neural Network (ANN)

## Overview
**[Kaggle Link](https://www.kaggle.com/code/nazliyusufoglu/notebook75ff098577)**
This project implements an **Artificial Neural Network (ANN)** to classify images of fish species. The dataset consists of `.png` images of various fish species organized into directories. The goal is to preprocess the data, train the model using TensorFlow and Keras, and evaluate its performance through various metrics.

## Project Structure
The project is divided into three main phases:
1. **Data Preprocessing**
2. **Model Training**
3. **Model Evaluation**

---

## 1. Data Preprocessing

### Steps:
1. **Image Loading**: 
   The images are loaded and resized to `150x150` pixels. Normalization is performed to scale pixel values between `0` and `1`.

   ```python
   def process_images(image_paths):
       images = []
       for path in image_paths:
           img = load_img(path, target_size=(150, 150))  # Resize images
           img = img_to_array(img)  # Convert to array
           img /= 255.0  # Normalize pixel values
           images.append(img)
       return np.array(images)
