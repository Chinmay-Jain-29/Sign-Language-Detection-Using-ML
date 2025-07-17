import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import array
Data_type = int
data_dict=pickle.load(open('./data.pickle','rb'))
# print(data_dict['data'])
# print(data_dict['labels'])
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])
data_list = data_dict['data']
data_labels = data_dict['labels']
# Find the maximum length of arrays in the list
max_length = max(len(arr) for arr in data_list)

# Pad sequences to the maximum length
padded_data = [np.pad(arr, (0, max_length - len(arr))) for arr in data_list]

max_length1 = max(len(arr) for arr in data_labels)

# Pad sequences to the maximum length
padded_labels = [np.pad(arr, (0, max_length - len(arr))) for arr in data_labels]

# Convert to NumPy array
data = np.asarray(padded_data)
labels = np.asarray(padded_labels)
print(data)
print(labels)
# print(labels)
#
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#
model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()