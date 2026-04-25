print("Logistic Regression Advertisement Click Predictor")

import pandas as pd

data = pd.read_csv("ad_click_dataset.csv")

#Data Prep: Clean the dataset
print(data.isnull().sum()) #Based on this line's output, I now realised that this dataset has NULL values for some features => need to fix it else would cause error when Sklearn LogisticRegression model training


data = data.fillna(data.mean(numeric_only=True)) #fill the NULL cells with mean (but only for numerical features)
data = data.dropna().reset_index(drop=True) #drop the rows with NULL cells (rows which contain NULL for the non-numerical features e.g. "device_type" of Person A is NULL (unknown) => we have no choice but to drop Person A's entire row from dataset cos Person A can't be used for model training) [Note: data.dropna() returns a dataframe, rmbr to set to data using "data = data.dropna().reset_index(drop=True)"]


print(data.head)
print(data.columns)
print(data.info())
print(data.describe())


#remove 'id', 'full_name' features since they are completely useless for our model
data = data.drop('id', axis=1)
data = data.drop('full_name', axis=1)


#Scikit Learn cannot understand text, they expect numbers so use One-Hot Encoding to convert text features to numbers
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()


encoded = encoder.fit_transform(data[['gender']])
encoded_dataframe = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
data = data.drop('gender', axis=1)
data = pd.concat([data, encoded_dataframe], axis=1)

encoded = encoder.fit_transform(data[['device_type']])
encoded_dataframe = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
data = data.drop('device_type', axis=1)
data = pd.concat([data, encoded_dataframe], axis=1)

encoded = encoder.fit_transform(data[['ad_position']])
encoded_dataframe = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
data = data.drop('ad_position', axis=1)
data = pd.concat([data, encoded_dataframe], axis=1)

encoded = encoder.fit_transform(data[['browsing_history']])
encoded_dataframe = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
data = data.drop('browsing_history', axis=1)
data = pd.concat([data, encoded_dataframe], axis=1)

encoded = encoder.fit_transform(data[['time_of_day']])
encoded_dataframe = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
data = data.drop('time_of_day', axis=1)
data = pd.concat([data, encoded_dataframe], axis=1)


#Be careful of Common Bug: The X, Y should be based on the post-onehotencoded "data" dataframe, NOT before onehotencoding the "data". Hence, the below 2 lines occur AFTER onehotencoding finishes
X = data.drop('click', axis=1) #X (input) is all the features in data, excluding the 'click' feature
Y = data['click']

print(Y.value_counts()) #observe the output of this line to check for imbalanced dataset [e.g. Clicks(1) >> Non-clicks(0)]


print(data.columns)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


#Important for any ML Algorithms that measure distance from answer: Scale the feature to make all data value be of similar values (prevent domination by any high-value feature in the Logistic Regression math addition equation)
#.fit() => learn
#.transform => apply to the 2D DataFrame
#.fit_transform => learn + apply to the 2D DataFrame
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']]) 


#Train the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced') #Create Logisitic Regression Model [Btw, class_weight='balanced' is important for handling class imbalance by making the model automatically assign higher importance (proportionally greater weight value) to the minority classes than majority classes (proportionally smaller weight value)]
model.fit(X_train, Y_train)


#Make predictions
predictions = model.predict(X_test)

#Evaluate Model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

#Confusion Matrix (tells you true positives, false postives, ...)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))

#Classification Report (in-depth analysis of how ur model predictions fared)
from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))
                                                       
