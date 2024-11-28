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
