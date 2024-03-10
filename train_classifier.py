import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Assuming 'data' is a list of sequences
data = data_dict['data']

# Pad sequences to the same length
data_padded = pad_sequences(data, dtype='float32', padding='post')  

# Reshape the 3D array to 2D
data_padded_2d = data_padded.reshape(data_padded.shape[0], -1)

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data_padded_2d, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
