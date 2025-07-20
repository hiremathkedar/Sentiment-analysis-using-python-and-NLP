# Sentiment-analysis-using-python-and-NLP

This repository contains a simple Natural Language Processing (NLP) project that performs **sentiment analysis** on the NLTK movie reviews dataset using a **TF-IDF vectorizer** and a **Multinomial Naive Bayes classifier**.

## 📝 Project Description

The script:

- Loads the NLTK movie reviews dataset
- Prepares and vectorizes the text data
- Trains a Multinomial Naive Bayes model
- Evaluates model performance with accuracy, classification report, and confusion matrix

## 📂 Files

- `nlp.py`: Main Python script for data loading, preprocessing, training, and evaluation
- `products.csv`: *(Uploaded but not used in this script; consider removing or updating your code to integrate it if needed)*

## 🚀 Getting Started

### Clone the repository

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

Install requirements
Ensure you have Python 3 installed. Install the required libraries using:

pip install pandas numpy nltk scikit-learn

Run the script
python nlp.py


📊 Output
The script will display:
Sample data preview
Model accuracy on the test set
Classification report (precision, recall, f1-score)
Confusion matrix

🛠️ Libraries Used
pandas: data manipulation
numpy: numerical operations
nltk: text processing and dataset
scikit-learn: vectorization, model building, and evaluation

💡 Future Improvements
Implement sentiment analysis on a custom dataset
Compare different models (Logistic Regression, SVM, etc.)
Deploy as an API or integrate with a frontend

🤝 Contributing
Feel free to fork this repository, improve the code, and submit a pull request.
