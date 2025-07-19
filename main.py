
  # main.py

# Import necessary libraries and modules
from sklearn.datasets import load_iris                      # Built-in Iris dataset from scikit-learn
from sklearn.model_selection import train_test_split        # For splitting data into train/test sets
from sklearn.ensemble import RandomForestClassifier         # Random Forest ML model
from sklearn.metrics import accuracy_score                  # To evaluate model accuracy
import pandas as pd                                         # For data manipulation using DataFrames and Series

# Load the Iris dataset
iris = load_iris()                                          # Loads a dictionary-like object with data and target

# Convert data to a pandas DataFrame for easier handling
X = pd.DataFrame(iris.data, columns = iris.feature_names)   # Feature matrix (4 columns of flower measurements)
Y = pd.Series(iris.target)                                  # Target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

# Splits the data into training and testing sets (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)             # random_state ensures reproducibility
                                                     
# Create a Random Forest Classifier model
model = RandomForestClassifier()                            # Using a default parameters

# Train the model on the training data
model.fit(X_train, Y_train)

# Use the trained model to make predictions on the test data
Y_pred = model.predict(X_test)

# Evaluate the model performance using accuracy score
accuracy = accuracy_score(Y_test, Y_pred)                   # Compares true vs predicted labels

# Print the accuracy of the model, rounded to 2 decimal places
print(f"Model Accuracy: {accuracy: .2f}")     

# Evaluate accuracy 
accuracy = accuracy_score(Y_test, model.predict(X_test))
print("Model accuracy:", round(accuracy, 2))               # e.g. - 0.97

# Make a prediction
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns = iris.feature_names)                            # Sepal & petal measurements
predicted_class = model.predict(sample)[0]
predicted_label = iris.target_names[predicted_class]
print("Prediction for input:\n", sample)

