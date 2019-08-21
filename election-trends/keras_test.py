import json
import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import optimizers

def absoluteFilePaths(directory = "./json-data"):
	for dirpath,_,filenames in os.walk(directory):
		if ('.DS_Store' in filenames):
			filenames.remove('.DS_Store')
		for f in filenames:
			yield os.path.abspath(os.path.join(dirpath, f))





def import_json_files(pathlist):
	trainx = []
	trainy = []


	for p in pathlist:


		with open(p, 'r') as f:
			train = json.load(f)

		data = list(train.values())
		data_int = [int(e) for e in data[1:]]
		x_train = data_int[1:len(data_int)-1]
		trainx.append(x_train)

		y_train = data_int[len(data_int)-1:]
		trainy.append(y_train)

	return np.array(trainx), np.array(trainy)





def makemodel2():

	print ("Making model")
	model = Sequential()

	BatchNormalization(
	axis=-1, momentum=0.99,
	epsilon=0.001,
	center=True,
	scale=True,
	beta_initializer='zeros',
	gamma_initializer='ones',
	moving_mean_initializer='zeros',
	moving_variance_initializer='ones',
	beta_regularizer=None,
	gamma_regularizer=None,
	beta_constraint=None,
	gamma_constraint=None)

	model.add(Dense(units=5, activation='relu', input_dim=5))
	# model.add(Dropout(0.3))
	model.add(Dense(units=20, activation='relu'))
	# model.add(Dropout(0.3))
	model.add(Dense(units=10, activation='relu'))
	# model.add(Dropout(0.3))
	model.add(Dense(units=5, activation='relu'))

	model.add(Dense(activation='softmax', output_dim=1))

	adam = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

	model.compile(loss='mean_squared_error',
			  optimizer=adam,
			  metrics=['accuracy'])

	return model



def trainmodel(x_train, y_train, pathtojson = './models/model1.json', pathtoh5 = './models/model1.h5'):
	print ("Training model")

	trainedmodel = makemodel2()
	trainedmodel.fit(x_train, y_train, epochs=10, batch_size=None)

	print ("Saving model")

	model_json = trainedmodel.to_json()
	with open(pathtojson, "w") as json_file:
		json_file.write(model_json)
	trainedmodel.save_weights(pathtoh5)
	print("Saved model to disk")

	# score, acc = trainedmodel.evaluate(x_test, y_test, batch_size=16)
	# print ("Scores for Test set: {}".format(score))
	# print ("Accuracy for Test set: {}".format(acc))

	return trainedmodel


def testmodel(pathtojson, pathtoh5, data, labels ):
	print ("Testing model")

	print("using model: {}".format(pathtojson))

	# load json and create model
	json_file = open(pathtojson, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights(pathtoh5)
	print("Loaded model from disk")

	loaded_model.compile(loss='categorical_crossentropy',
						 optimizer='adam',
						 metrics=['accuracy'])
	print(data.shape)

	labels = to_categorical(labels)
	score, acc = loaded_model.evaluate(data, labels, batch_size=16)
	print ("Scores for Test set: {}".format(score))
	print ("Accuracy for Test set: {}".format(acc))


def loadmodel(pathtojson, pathtoh5):

	json_file = open(pathtojson, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights(pathtoh5)

	return loaded_model




def loadandpredict(test_data, pathtojson='./models/model1.json', pathtoh5='./models/model1.h5'):

	print("using model: {}".format(pathtojson))

	# load json and create model
	json_file = open(pathtojson, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	#load weights into new model

	loaded_model.load_weights(pathtoh5)
	print("Loaded model from disk")


	loaded_model.compile(loss='mean_squared_error',
				  optimizer='adam',
				  metrics=['accuracy'])
	# print ("compiled the loaded model with cat. cross entropy with adam optim...")

	# print ("shape of data {}".format(data.shape))

	classes = loaded_model.predict(test_data)

	return classes



if __name__ == '__main__':

	x_train, y_train = import_json_files(absoluteFilePaths())
	test_data = import_json_files(absoluteFilePaths('./test'))

	# trained_model = trainmodel(x_train, y_train)
	prediction = loadandpredict(test_data[0])
	print(prediction)



