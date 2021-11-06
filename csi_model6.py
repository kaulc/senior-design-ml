import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pprint

seed = 7
numpy.random.seed(seed)

# PREPROCESSING
# load dataset
dataframe = pandas.read_csv('train_multi.csv', header=None)
dataset = dataframe.values
X = dataset[:,0:53].astype(float)
Y = dataset[:,53]

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
	model.add(Dense(6, activation='softmax')) # normalizes output to 1

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# TRAIN, PREDICT
# classifier wrapper 
estimator = KerasClassifier(build_fn=baseline_model, epochs=250, batch_size=5, verbose=0)

# split dataset into training and validation set (just for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=seed)

# fit model
estimator.fit(X_train, Y_train)

# predict
predictions = estimator.predict(X_test)
predictions = encoder.inverse_transform(predictions)

# get model probablilities for each class
predictions_class = estimator.predict_proba(X_test)

# use decimal notion not scientific notation
numpy.set_printoptions(suppress=True)

print(predictions)
print(predictions_class)
