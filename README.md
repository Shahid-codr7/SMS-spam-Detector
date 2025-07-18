# SMS Spam Detector

This repository hosts an SMS Spam Detector application that leverages **machine learning** and **natural language processing (NLP)** techniques to classify SMS messages as spam or legitimate. The project achieves an impressive accuracy of over 95% on a dataset of 5,000+ SMS messages.

## Features

- **Machine Learning Model**: Implements a supervised machine learning model capable of distinguishing spam messages from legitimate ones.
- **Natural Language Processing (NLP)**: Utilizes NLP techniques for preprocessing and feature extraction from text data.
- **High Accuracy**: Achieves more than 95% accuracy on the test dataset.
- **Extensive Dataset**: Trained on a dataset of over 5,000 SMS messages, ensuring robustness and reliability.
- **Web Deployment**: The application is deployed for live testing and can be accessed [here](https://huggingface.co/spaces/Shahid-codr7/SMS-spam-detector)).

## Process Overview

1. **Dataset Preparation**:
   - The dataset consists of labeled SMS messages (spam or legitimate).
   - URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
   - The data is split into training and testing sets to evaluate model performance.

2. **Text Preprocessing**:
   - Removes stop words and punctuations.
   - Converts text to lowercase for normalization.
   - Tokenizes the text into individual words.

3. **Feature Extraction**:
   - Extracts meaningful features from the text using methods such as **Term Frequency-Inverse Document Frequency (TF-IDF)**.
   - Converts text into numerical vectors suitable for machine learning algorithms.

4. **Model Training**:
   - Several machine learning algorithms are tested (e.g., **Naive Bayes**, **Logistic Regression**, **Support Vector Machines (SVM)**).
   - Hyperparameter tuning is performed to optimize the chosen model.

5. **Evaluation**:
   - The model is evaluated on a test set using metrics like accuracy & precision.
   - A confusion matrix is created to visualize the performance.

6. **Deployment**:
   - The trained model is deployed on a web platform for real-time spam detection. You can access the live application [here](https://huggingface.co/spaces/Shahid-codr7/SMS-spam-detector)).

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Shahid-codr7/SMS-spam-Detector.git
   cd SMS-spam-Detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Train or test the model by following the steps in the provided notebook.

## Results

The model achieves:
- **Accuracy**: Over 95%
- **Precision**: High precision for spam detection.
- **Recall**: High recall for legitimate messages.

## Dataset

The dataset used for training and evaluation contains over 5,000 SMS messages with labels (`spam` or `ham`).

## Technologies Used

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, NLTK
- **Tools**: Jupyter Notebook

## Future Improvements

- Expand the dataset to include more diverse SMS messages.
- Experiment with deep learning models like LSTMs or transformers for further improvements.
- Enhance the web application for better user experience and scalability.

---

Let me know if any further updates or refinements are needed!
