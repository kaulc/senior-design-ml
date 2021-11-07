import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import sys

seed = 7
numpy.random.seed(seed)

# PREPROCESSING
# load dataset
#dataframe = pandas.read_csv("TrainingDataSet.csv", header=0)
dataframe = pandas.read_excel("TrainingDataSet.xlsx")
dataset = dataframe.values
X_mag = dataset[:,1:54].astype(float)
X_phase = dataset[:,54:107].astype(float)
Y = dataset[:,0]

# assign integer to each class
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# one hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)


# CREATE MODEL
def baseline_model():
	model = Sequential()
	model.add(Dense(24, input_dim=53, activation='relu'))
	model.add(Dense(7, activation='softmax')) # normalizes output to 1

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# TRAIN MODEL
# classifier wrapper 
estimator = KerasClassifier(build_fn=baseline_model, epochs=250, batch_size=5, verbose=0)
#estimator_p = KerasClassifier(build_fn=baseline_model, epochs=250, batch_size=5, verbose=0)

# split dataset into training and validation set *** just for testing *** 
X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(X_mag, dummy_y, test_size=0.1, random_state=seed)
#X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(X_phase, dummy_y, test_size=0.1, random_state=seed)

# fit model
estimator.fit(X_train_m, Y_train_m) #mag
#estimator_p.fit(X_train_p, Y_train_p) #phase 

# PREDICT
 # *** replace with real-time data input *** 
predictions_m = estimator.predict(X_test_m)
#predictions_p = estimator_p.predict(X_test_p)

# transform predictions from integers to class names
predictions_m = encoder.inverse_transform(predictions_m)
#predictions_p = encoder.inverse_transform(predictions_p)

# get model probablilities for each class
predictions_classes_m = estimator.predict_proba(X_test_m)
#predictions_classes_p = estimator_p.predict_proba(X_test_p)


# PRINT
# use decimal notion not scientific notation
numpy.set_printoptions(suppress=True)

# don't truncate arrays
numpy.set_printoptions(threshold=sys.maxsize)

print(predictions_m)
print(predictions_classes_m)
#print(predictions_p)
#print(predictions_classes_p)
