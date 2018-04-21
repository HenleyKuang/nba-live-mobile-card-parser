from sklearn import datasets
from sklearn.datasets.base import Bunch
from skimage.feature import hog
from sklearn import preprocessing
from optparse import OptionParser
import cv2
import os
import numpy as np

digits = Bunch()
digits.data = []
digits.target = []
digits.target_names = []

# Get the path of the training set
parser = OptionParser()
parser.add_option("-n", dest="classify_name", help="name of classification", action="store")
(options, args) = parser.parse_args()

classify_name = options.classify_name

parent_path = classify_name
for category in os.listdir(parent_path):
    full_category_path = os.path.join(parent_path, category)
    if not os.path.isdir(full_category_path):
      continue
    for file in os.listdir(full_category_path):
        full_file_path = os.path.join(full_category_path, file)
        im = cv2.imread(full_file_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (30, 30), interpolation=cv2.INTER_AREA)
        im = np.array(im, 'float64')
        digits.data.append(im)
        digits.target.append(category)
    if category not in digits.target_names:
      digits.target_names.append(category)

# Extract the features and labels
features = np.array(digits.data, 'int16')
labels = np.array(digits.target, 'string')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((30,30)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print "training..."
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(hog_features, labels)
# Save the classifier
from sklearn.externals import joblib
joblib.dump((clf, pp), 'PKL/%s.pkl' % classify_name, compress=3)

print "testing..."
correct = 0
incorrect = 0
incorrect_list = {}
for category in os.listdir(parent_path):
    full_category_path = os.path.join(parent_path, category)
    if not os.path.isdir(full_category_path):
      continue
    for file in os.listdir(full_category_path):
      full_file_path = os.path.join(full_category_path, file)
      pic_data = cv2.imread(full_file_path)
      pic_data = cv2.cvtColor(pic_data, cv2.COLOR_BGR2GRAY)
      pic_data = cv2.resize(pic_data, (30, 30), interpolation=cv2.INTER_AREA)
      pic_data = np.array(pic_data, 'int16')
      pic_hog_fd = hog(pic_data.reshape((30,30)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
      pic_hog_fd = pp.transform(np.array([pic_hog_fd], 'float64'))
      print full_file_path
      prediction = clf.predict(pic_hog_fd)[0]
      print prediction
      if prediction == category:
        correct += 1
      else:
        incorrect += 1
        incorrect_list.update({full_file_path: prediction})


print "Correct: %s" % correct
print "Incorrect: %s" % incorrect
accuracy = 100.0 if incorrect == 0 else (correct * 1.0/(correct + incorrect)) * 100
print "Accuracy: %s" % accuracy
print incorrect_list