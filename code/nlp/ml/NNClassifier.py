<<<<<<< HEAD
import matplotlib.pyplot as plt
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neural_network import MLPClassifier

# set up lists
tweets_train = []
sentiment_train = []
tweets_test=[]
sentiment_test = []

# read in the csv file and save variables in respective list
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

# Initialize and train the Neural Network model
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=42, verbose=True)
model.fit(tweets_train_vectorized, sentiment_train)

# Make predictions
predictions = model.predict(tweets_test_vectorized)

# Generate the confusion matrix
cm = confusion_matrix(sentiment_test, predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Explicitly show the plot
plt.show()

# Show accuracy
print("Accuracy:", accuracy_score(sentiment_test, predictions))
=======
import matplotlib.pyplot as plt
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neural_network import MLPClassifier

tweets = []
sentiment = []
with open('./data/training.1600000.processed.noemoticon.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Iterate over rows
    for row in reader:
        tweets.append(row[5])
        sentiment.append(row[0])

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size=0.00125, random_state=42)

# Initialize and train the Neural Network model
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=2, random_state=42, verbose=True)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Explicitly show the plot
plt.show()

# Show accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
>>>>>>> f5261a57c5ac9bb7b93abbeaab09f196abe5b8a8
