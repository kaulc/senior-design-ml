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
import matplotlib.pyplot as plt
import sys

seed = 7
numpy.random.seed(seed)


# PRINT
# use decimal notion not scientific notation
numpy.set_printoptions(suppress=True)

# don't truncate arrays
numpy.set_printoptions(threshold=sys.maxsize)

pandas.set_option("display.max_rows", None, "display.max_columns", None)


# PREPROCESSING
# read training set
dataframe = pandas.read_csv("Data1.csv")
dataset = dataframe.values
Xm = dataset[:,1:54].astype(float)
Xp = dataset[:,54:108].astype(float)
Y = dataset[:,0]

# check for NaN values
assert not numpy.any(numpy.isnan(Xm))
assert not numpy.any(numpy.isnan(Xp))

# assign integer to each class
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# one hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)

# CREATE MODEL
def baseline_model():
	model = Sequential()
	model.add(Dense(200, input_dim=53, activation='sigmoid'))
	model.add(Dense(400, activation='relu'))
	model.add(Dense(200, activation='relu'))
	model.add(Dense(7, activation='softmax')) # normalizes output to 1

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# TRAIN MODEL
# classifier wrapper 
estimatorM = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=200, verbose=1)
estimatorP = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=200, verbose=1)

# split dataset into training and validation set *** just for testing *** 
Xm_train, Xm_test, Ym_train, Ym_test = train_test_split(Xm, dummy_y, test_size=0.1, random_state=seed)
Xp_train, Xp_test, Yp_train, Yp_test = train_test_split(Xp, dummy_y, test_size=0.1, random_state=seed)

# fit model
histM = estimatorM.fit(Xm_train, Ym_train,  validation_data=(Xm_test, Ym_test)) #mag
histP = estimatorP.fit(Xp_train, Yp_train,  validation_data=(Xp_test, Yp_test)) #mag

# PREDICT
predictionsM = estimatorM.predict(Xm_test)
predictionsP = estimatorP.predict(Xp_test)

# transform predictions from integers to class names
predictionsM_inv = encoder.inverse_transform(predictionsM)
predictionsP_inv = encoder.inverse_transform(predictionsP)

# get model probablilities for each class
predictionsM_classes = estimatorM.predict_proba(Xm_test)
predictionsP_classes = estimatorP.predict_proba(Xp_test)

Ym_test = encoder.inverse_transform(numpy.argmax(Ym_test, axis=1))
Yp_test = encoder.inverse_transform(numpy.argmax(Yp_test, axis=1))


# PRINT

print("----------MAG-------------")
print("Prediction\tReal\t\tConfidence\tFlagCorrect\tIndex")
output = ""
count = 0
for i in range(len(predictionsM)):
	print(predictionsM_inv[i], end = "\t\t")

	print(Ym_test[i], end = "\t\t")
	
	print(predictionsM_classes[i][predictionsM[i]], end = "\t\t")
	
	if (predictionsM_inv[i] == Ym_test[i]): 
		count = count + 1
		print("*", end="\t\t")
	else:
		print(" ", end="\t\t")

	print(i)

accuracy = count / len(predictionsM)
print("\nAccuracy out of 1 = " + str(accuracy))


print("----------PHASE-------------")
print("Prediction\tReal\t\tConfidence\tFlagCorrect\tIndex")
output = ""
count = 0
for i in range(len(predictionsP)):
	print(predictionsP_inv[i], end = "\t\t")

	print(Yp_test[i], end = "\t\t")
	
	print(predictionsP_classes[i][predictionsP[i]], end = "\t\t")
	
	if (predictionsP_inv[i] == Yp_test[i]): 
		count = count + 1
		print("*", end=" ")	
	else:
		print(" ", end="\t\t")

	print(i)

accuracy = count / len(predictionsP)
print("\nAccuracy out of 1 = " + str(accuracy))


print(histM.history.keys())
# summarize history for accuracy
plt.plot(histM.history['accuracy'])
plt.plot(histM.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(histM.history['loss'])
#plt.plot(histM.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()