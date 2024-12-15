
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import requests
from io import StringIO
from zipfile import ZipFile
import io

# Step 1: Load dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)

# Load the content as a zip file and read it
with ZipFile(io.BytesIO(response.content)) as z:
    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])

# Map labels to binary values: 'spam' -> 1, 'ham' -> 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Step 2: Text vectorization using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])  # Transform text data into feature vectors
y = df['label']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set and evaluate the model
y_pred = model.predict(X_test)
print("Model Training Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: User Input for Spam Prediction with Probability Output
while True:
    user_input = input("Enter a message to check if it's spam (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    # Transform the user input into the same feature space
    user_input_vector = vectorizer.transform([user_input])
    
    # Get the probability predictions
    spam_probability = model.predict_proba(user_input_vector)[0][1]  # Probability of being spam
    ham_probability = model.predict_proba(user_input_vector)[0][0]   # Probability of being ham

    # Print the prediction with probabilities
    if spam_probability > 0.5:
        print(f"The message is likely spam with a {spam_probability * 100:.2f}% probability.")
    else:
        print(f"The message is likely not spam with a {ham_probability * 100:.2f}% probability.")
