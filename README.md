# 🌐 LinguaDetect AI — Language Detection Using NLP

## 📌 Project Overview

LinguaDetect AI is a Natural Language Processing (NLP) project that automatically detects the language of a given text.

The application accepts text from the user and predicts its language using **TF-IDF character n-gram feature extraction** and a **Multinomial Naive Bayes classification algorithm**.

The project is developed using **Python** and **Streamlit** to provide an interactive web interface.

## 🎯 Objectives

- Detect the language of user-provided text.
- Apply NLP techniques for text feature extraction.
- Convert text into numerical features using TF-IDF.
- Use character n-grams to capture language-specific patterns.
- Classify text using Multinomial Naive Bayes.
- Display the predicted language and probability/confidence.
- Provide a simple and user-friendly web interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NLP | Text processing and language detection |
| Scikit-learn | TF-IDF and machine learning model |
| TF-IDF | Text feature extraction |
| Character N-Grams | Capture character patterns |
| Multinomial Naive Bayes | Language classification |
| Streamlit | Web application interface |

## 🌍 Supported Languages

- 🇬🇧 English
- 🇮🇳 Tamil
- 🇫🇷 French
- 🇪🇸 Spanish
- 🇩🇪 German

## 🔄 Project Workflow

```text
User Input
    ↓
Character N-Grams
    ↓
TF-IDF Feature Extraction
    ↓
Numerical Feature Vector
    ↓
Multinomial Naive Bayes
    ↓
Language Prediction
    ↓
Probability / Confidence
```

## 🧠 NLP Implementation

The main NLP component is:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5)
)
```

### Character N-Grams

The project uses 2 to 5 character n-grams.

For example, for `hello`:

```text
2-grams → he, el, ll, lo
3-grams → hel, ell, llo
4-grams → hell, ello
5-grams → hello
```

Character patterns are useful for language detection because different languages have different combinations and sequences of characters.

## 📊 TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical values that the machine-learning model can understand.

In this project, TF-IDF is applied to character n-grams.

```python
X = vectorizer.fit_transform(texts)
```

`fit_transform()` learns the character patterns from the training data and converts the text into numerical TF-IDF feature vectors.

## 🤖 Machine Learning Algorithm

### Multinomial Naive Bayes

The project uses:

```python
model = MultinomialNB()
```

The model is trained using:

```python
model.fit(X, languages)
```

Here:

- `X` → TF-IDF numerical features
- `languages` → corresponding language labels

For new text:

```python
user_vector = vectorizer.transform([user_text])
prediction = model.predict(user_vector)[0]
```

The model can also provide probabilities for the language classes:

```python
probabilities = model.predict_proba(user_vector)[0]
```

## 💡 Why Character N-Grams?

Character n-grams are suitable for language detection because they capture patterns inside words without depending entirely on complete words.

The project uses:

```python
ngram_range=(2, 5)
```

This provides a balance between small character patterns and longer character combinations.

## 🖥️ Application Features

- Text input through a Streamlit interface
- Automatic language detection
- Predicted language display
- Probability/confidence display
- Probability distribution for supported languages
- Interactive and simple user interface

## 📂 Project Structure

```text
Language-Detection-NLP/
│
├── app.py
├── requirements.txt
└── README.md
```

### `app.py`

Contains the training data, TF-IDF vectorizer, character n-gram configuration, Multinomial Naive Bayes model, prediction logic, and Streamlit interface.

### `requirements.txt`

Contains the Python libraries required to run the project.

Example:

```text
streamlit
scikit-learn
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project folder

```bash
cd Language-Detection-NLP
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

The application will normally be available at:

```text
http://localhost:8501
```

## 🧪 Example

### Input

```text
I am learning artificial intelligence.
```

### Output

```text
Detected Language: English
```

The application also displays the probability calculated by the model.

## 🔑 Key Concepts

### NLP
Natural Language Processing allows computers to process and analyze human language.

### TF-IDF
A feature extraction technique that converts text into numerical feature values.

### Character N-Gram
A sequence of consecutive characters of a specified length.

### Multinomial Naive Bayes
A probability-based classification algorithm used to classify the extracted text features into language classes.

### Streamlit
A Python framework used to create the interactive web application.

## ⚠️ Limitations

- The current project uses a relatively small training dataset.
- It supports only five languages.
- Very short text may be difficult to classify.
- Similar languages can sometimes have similar character patterns.
- The displayed probability is the model's predicted probability for a particular input; it is not the overall model accuracy.

## 🔮 Future Enhancements

- Add more languages.
- Use a larger language dataset.
- Improve text preprocessing.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Add confusion matrix visualization.
- Allow users to upload text files.
- Deploy the application online.
- Experiment with other NLP and machine-learning models.

## 👩‍💻 Project Summary

LinguaDetect AI demonstrates how NLP and machine learning can be combined to solve a practical language classification problem.

The core implementation is:

```text
Character N-Grams
        ↓
      TF-IDF
        ↓
Multinomial Naive Bayes
        ↓
Language Prediction
```

This project demonstrates NLP-based text feature extraction, machine-learning classification, and interactive application development using Streamlit.
