#importing the libraries needed:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#now we have to load in training the data from question/4:
#load the data set into a data frame first:
data_frame = pd.read_csv("kidney_disease.csv")

#we have to replaces the values are not there:
data_frame.replace('?', pd.NA, inplace=True)

#now we have to separate the x (features) and the y (label):
x = data_frame.drop("classification", axis=1)
y = data_frame["classification"]

#converting the categorical variables into numeric and also filling in the missing numeric values with column means:
x = pd.get_dummies(x)
x = x.fillna(x.mean())

#now we can split into the testing and training data:
x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42
)

#training the KNN models:
#assinging the k values and creating an empty list for the final accuracy values:
k_values = [1,3,5,7,9]
final_accuracy_results = []

#looping through the values, testing each training model and saving the results to find the highest test accuracy:
for k_value in k_values:
    knn_model = KNeighborsClassifier(n_neighbors = k_value)
    knn_model.fit(x_training, y_training)

    predictions = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    final_accuracy_results.append(accuracy)

#now we have to display the results table:
print("k value     Test Accuracy")
print("==========================")
for index in range(len(k_values)):
    print(k_values[index], "         ", final_accuracy_results[index])

#identifiying the best k we have:
best_accuracy = max(final_accuracy_results)
best_k_value = k_values[final_accuracy_results.index(best_accuracy)]

#prinint the best k value and best accuracy results:
print("\nThe best k value is:", best_k_value)
print("The highest test accuracy is:", best_accuracy)

#=============================================================================
#Explanation:
'''
When the k value is changed the flexibility of the KNN model is effected. Moreover, small
k values may cause overfitting because the model will memorize the training data and will then 
not be able to generalize the new data. Very large values of k may cause underfitting due to
the model ignoring the important patterns in the data. 
'''
