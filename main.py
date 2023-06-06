from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt

def load_dataset(filename, split_ratio):
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable

  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    return list(zip(X_train, y_train)), list(zip(X_test, y_test))

def knn_classification(training_set, test_set, k):
    X_train, y_train = zip(*training_set)
    X_test, y_test = zip(*test_set)


    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(X_train, y_train)

 
    predictions = classifier.predict(X_test)

    return predictions

def get_accuracy(test_set, predictions):
    y_test = list(zip(*test_set))[1]
    return accuracy_score(y_test, predictions) * 100.0


filename = Path('D:/Task 1- Iris flower classification/Iris-Flower-Classification-Dataset-main/IRIS.csv')
split_ratio = 0.2  # 20% for testing
training_set, test_set = load_dataset(filename, split_ratio)

k = 3
predictions = knn_classification(training_set, test_set, k)

accuracy = get_accuracy(test_set, predictions)
print(f"Accuracy: {accuracy:.2f}%")

features, labels = zip(*training_set)

plt.scatter([feature[0] for feature in features], [feature[1] for feature in features], c=labels, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Flower Classification')
plt.show()
