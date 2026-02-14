#importing pandas and sklearn:
import pandas as pd
from sklearn.model_selection import train_test_split

#loading in the dataset
data_frame = pd.read_csv("kidney_disease.csv")

#creating a feature matrix that is for all the columns except CKD:
x = data_frame.drop("classification", axis=1)

#creating a label vector y, a CKD column:
y = data_frame["classification"]

#now well split up the dataset: (into 70% training and then 30% testing)
x_training, x_test, y_training, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42
)

#now we can print the shapes to confirm the split:
print("The training features shape is:", x_training.shape)
print("The testing features shape is:", x_test.shape)
print("The training labels shape is:", y_training.shape)
print("The testing labels shape is:",y_test.shape)

#========================================================================
#Explanation:
'''
A model should not be trained and tested on the same data because it will lead to inaccurate results.
Instead different data should be used for the training and testing. If it is done with the same data then
the model could memorized the training examples instead of learning the general patterns. Therefore, overfitting
would result. 
The purpose of the testing set is to be able to measure how well the model is able to generalize new data that has not been 
seen before which will also be able to prevent overfitting from occurring. Therefore, a unbiased estimate of the models actual performance is provided.
'''