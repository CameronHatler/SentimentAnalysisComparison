<<<<<<< HEAD
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import csv

# Open the CSV file
tweets_train = []
sentiment_train = []
tweets_test=[]
sentiment_test = []
with open('./data/training.1200000.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Iterate over rows
    for row in reader:
        tweets_train.append(row[5])
        sentiment_train.append(row[0])

with open('./data/test.320000.csv', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        tweets_test.append(row[5])
        sentiment_test.append(row[0])
    

# Convert text data to feature vectors
vectorizer = CountVectorizer()
tweets_train_vectorized = vectorizer.fit_transform(tweets_train)
tweets_test_vectorized = vectorizer.transform(tweets_test)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(tweets_train_vectorized, sentiment_train)

# Make predictions
predictions = model.predict(tweets_test_vectorized)

# Generate the confusion matrix
cm = confusion_matrix(sentiment_test, predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

plt.show()

# Evaluate the model
print("Accuracy:", accuracy_score(sentiment_test, predictions))

=======
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import csv

# Open the CSV file
tweets = []
sentiment = []
tweets_training=[]
sentiment_training = []
with open('./data/training.1600000.processed.noemoticon.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Iterate over rows
    counter = 1
    for row in reader:
        if counter <= 1000:
            tweets_training.append(row[5])
            sentiment_training.append(row[0])
        elif counter <= 1590000:
            tweets.append(row[5])
            sentiment.append(row[0])
        else:
            tweets_training.append(row[5])
            sentiment_training.append(row[0])


# Convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.00125, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

plt.show()

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))

>>>>>>> f5261a57c5ac9bb7b93abbeaab09f196abe5b8a8
