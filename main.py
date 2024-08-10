import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Define a function to load data from an online source
def load_data():
    # Load the Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data

# Define a function to fit, predict, and evaluate a model
def model_workflow(data):
    # Split data into features and target
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define a Decision Tree Classifier
    model = DecisionTreeClassifier()
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=iris.target_names)
    
    return accuracy, report

# Main execution
if __name__ == "__main__":
    # Load the dataset
    data = load_data()
    
    # Perform modeling
    accuracy, report = model_workflow(data)
    
    # Print results
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
