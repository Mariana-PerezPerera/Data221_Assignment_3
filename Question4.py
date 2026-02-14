#importing the libraries that are needed:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#now we have to initialize the KNN classifier with having k=5 (the number of neighbors)
knn = KNeighborsClassifier(n_neighbors = 5)

#need the training data, using the code from question 3:
#load in the data set:
data_frame = pd.read_csv("kidney_disease.csv")
#we have to replace the blanks with actual values(NaN)
data_frame.replace('?', pd.NA, inplace=True)
#we also have to make sure to convert all of the columns into numeric where we can:
# data_frame = data_frame.apply(pd.to_numeric, errors = 'coerce')

#creating a feature matrix that is for all the columns except CKD:
x = data_frame.drop("classification", axis=1)
#converting any categorical variables to numeric:
x = pd.get_dummies(x)
# we have to fill the missing numeric values with column means as well:
x = x.fillna(x.mean())

#creating the y label vector:
y = data_frame["classification"]

#now splitting up the data set so we can get the training data:
x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42
)


#now we have to train the model,and we will use the training data from question 3:
knn.fit(x_training, y_training)

#now we will make a prediction on the testing data:
testing_data_prediction_for_y = knn.predict(x_test)

#now we will compute the confusion matrix:
confusionMatrix = confusion_matrix(y_test, testing_data_prediction_for_y)
print(confusionMatrix)

#computing all the evaluation matrices:
accuracy= accuracy_score(y_test, testing_data_prediction_for_y)
precision= precision_score(y_test, testing_data_prediction_for_y, pos_label='ckd')
recall= recall_score(y_test, testing_data_prediction_for_y, pos_label='ckd')
f1 = f1_score(y_test, testing_data_prediction_for_y, pos_label='ckd')

#printing all the score we just computed above:
print("The accuracy is: ", accuracy)
print("The precision is: ", precision)
print("The recall is: ", recall)
print("The f1 score is: ", f1)

#=========================================================================
#Explanation:
'''
In the context of the kidney disease prediction the True Positive means that the model was 
able to correctly predict that a patient does have chronic kidney disease. Meanwhile, a True Negative in 
context of the dataset is the correct prediction that a patient does not have chronic kidney disease. 
Therefore, a False Positive is the model predicting that a patient does have chronic kidney disease when
they in fact do not have it, and a False Negative is the model predicting that a patient does not have 
chronic kidney disease when they actually do have it. 
Accuracy alone may not be enough to evaluate a classification model because the dataset itself could not be 
balanced. For example, if a majority of the patients do not have chronic kidney disease the model could end up
having a high accuracy because it could predict no chronic kidney disease most of the time.
The metric that is most important if a missing kidney disease case is very serious would be, the recall. It is 
the most important because recall measures how well the model is able to identify the actual chronic kidney disease 
cases while alos minimizing the false negatives.
'''