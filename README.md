# Twitter Sentiment Analysis 🐦💬  
**Minor Project – Exploratory Data Analysis & Classification**

## 📌 Project Overview  
This project analyzes **1.6M tweets** from the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) to perform:  
1. **Exploratory Data Analysis (EDA)** – understanding tweet distributions, trends, and relationships.  
2. **Preprocessing & Feature Engineering** – cleaning tweets, handling emojis/URLs, and encoding sentiment classes.  
3. **Classification** – training a **Logistic Regression model** to predict tweet sentiment (positive/negative).  

---

## 📂 Dataset  
- **Source**: Sentiment140 Kaggle dataset  
- **Size**: 1.6M tweets  
- **Columns**:  
  - `target`: Sentiment (0 = Negative, 4 = Positive, 2 = Neutral)  
  - `ids`: Tweet ID  
  - `date`: Date of posting  
  - `flag`: Query flag (not useful)  
  - `user`: Username of tweeter  
  - `text`: Tweet content  

---

## 🛠️ Steps Performed  

### **Task 1 – Exploratory Data Analysis (EDA)**  
✔ Checked for **missing values** (none found).  
✔ Verified **duplicates** (none found).  
✔ Analyzed **tweet posting patterns** by date.  
✔ Distribution of **positive vs. negative tweets**.  
✔ Correlation between **tweet length and sentiment** (negligible).  
✔ One-hot encoding for sentiment classes.  
✔ Preprocessed tweets by:  
- Removing special characters & URLs  
- Handling mentions (@usernames)  
- Encoding/removing emojis  

---

### **Task 2 – Classification**  
1. **Data Preprocessing**  
   - Cleaned text (lowercasing, punctuation removal, tokenization).  
   - Encoded emojis as text.  
   - Handled categorical labels using **one-hot encoding**.  

2. **Feature Engineering**  
   - Extracted **Bag-of-Words (BoW)** using `CountVectorizer`.  
   - Created **tweet length** feature.  

3. **Train-Test Split**  
   - 80% training, 20% testing.  

4. **Model Training**  
   - Algorithm: **Logistic Regression** (`max_iter=1000`)  
   - Achieved **~80% accuracy** on test data.  

5. **Model Evaluation**  
   - **Confusion Matrix**  
   - **Classification Report** (Accuracy, Precision, Recall, F1-Score)  

6. **Predictions**  
   - Example tweets classified correctly as positive/negative.  

---

## 📊 Results & Observations  
- Dataset had **balanced classes**: 800k positive & 800k negative tweets.  
- **Twitter activity peaked** around June 2009.  
- **Negligible correlation** between tweet length & sentiment.  
- Logistic Regression achieved **~80% accuracy** (baseline sentiment classifier).  
- Preprocessing (removing URLs, mentions, emojis) improved text clarity.  

---

## 🚀 Future Improvements  
- Implement **advanced models**: Naive Bayes, SVM, Random Forest, or Deep Learning (RNN, LSTM, BERT).  
- Perform **hyperparameter tuning** using Grid Search / Random Search.  
- Use **TF-IDF vectorization** instead of simple BoW.  
- Apply **word embeddings** (Word2Vec, GloVe, FastText).  
- Deploy as a **Flask/Streamlit web app** for real-time tweet sentiment prediction.  

---

## 🖥️ Tech Stack  
- **Python** 🐍  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Emoji, Re  
- **Dataset Source**: Kaggle – Sentiment140  

---

## 📌 How to Run  
```bash
# Clone repo
git clone <repo-url>
cd twitter-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebook in Google Colab / Jupyter
