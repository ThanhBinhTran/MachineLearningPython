from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a random forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Print the model's accuracy
    print(f'Model accuracy: {model.score(X_test, y_test)}')

if __name__ == '__main__':
    main()
