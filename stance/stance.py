import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import cycle
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":
	datapattern = 'data/abortion/*.data'
	data = glob.glob(datapattern)

	metapattern = 'data/abortion/*.meta'
	meta = glob.glob(metapattern)

	#print data
	X = []
	for i in range(len(data)):
		dataFile = open(data[i], 'r')
		line = dataFile.readline()
		#print line
		X.append(line)

	#print meta
	Y = []
	for i in range(len(meta)):
		metaFile = open(meta[i], 'r')
		line = metaFile.readline()
		line = metaFile.readline()
		line = metaFile.readline()
		print line
		print meta[i]
		line = line.split("Stance=")[1]
		y = int(line.split("\n")[0])
		Y.append(y)

	split = len(X) / 2
	Xtrain = X[:split]
	Ytrain = Y[:split]
	Xtest = X[split:]
	Ytest = Y[split:]

	# Create the cosine similarity SVM
	cos_pipe = svm.SVC(C=0.5, kernel=cosine_similarity, decision_function_shape='ovr')

	# Initialize the counter vectorizer and tfidf transformer with the correct params
	count_vect = CountVectorizer(lowercase=True, stop_words='english')
	tfidf_transformer = TfidfTransformer(sublinear_tf=True, norm='l2')
	tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2', lowercase=True, stop_words='english')

	# Create a pipeline for the cosine similarity SVM
	#cos_pipe = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', cos_clf), ])
	Xtrain = tfidf_vectorizer.fit_transform(Xtrain)
	Xtest = tfidf_vectorizer.transform(Xtest)


	# Fit the classifier and time the fitting
	cos_pipe = cos_pipe.fit(Xtrain, Ytrain)

	# Find the outputs of predict and decision_function
	cos_train_predict = cos_pipe.predict(Xtrain)
	cos_test_predict = cos_pipe.predict(Xtest)
	svm_y_test_score = cos_pipe.decision_function(Xtest)

	# Collect the accuracy, precision, and recall scores
	cos_accuracy_train = accuracy_score(Ytrain, cos_train_predict)
	cos_accuracy_test = accuracy_score(Ytest, cos_test_predict)

	cos_precision_train = precision_score(Ytrain, cos_train_predict, average='macro')
	cos_precision_test = precision_score(Ytest, cos_test_predict, average='macro')

	cos_recall_train = recall_score(Ytrain, cos_train_predict, average='macro')
	cos_recall_test = recall_score(Ytest, cos_test_predict, average='macro')

	# Create a pipeline for Naive Bayes
	#nb_pipe = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', MultinomialNB()), ])
	nb_pipe = MultinomialNB()

	# Fit the classifier and time the fitting
	nb_pipe = nb_pipe.fit(Xtrain, Ytrain)

	# Find the outputs of predict and predict_proba
	nb_y_test_score = nb_pipe.predict_proba(Xtest)
	nb_train_predict = nb_pipe.predict(Xtrain)
	nb_test_predict = nb_pipe.predict(Xtest)

	# Collect the accuracy, precision, and recall scores
	nb_accuracy_train = accuracy_score(Ytrain, nb_train_predict)
	nb_accuracy_test = accuracy_score(Ytest, nb_test_predict)

	nb_precision_train = precision_score(Ytrain, nb_train_predict, average='macro')
	nb_precision_test = precision_score(Ytest, nb_test_predict, average='macro')

	nb_recall_train = recall_score(Ytrain, nb_train_predict, average='macro')
	nb_recall_test = recall_score(Ytest, nb_test_predict, average='macro')

	# Output all scores
	print "Accuracy NB Train: " + str(nb_accuracy_train)
	print "Accuracy Cos Train: " + str(cos_accuracy_train)

	print "Accuracy NB Test: " + str(nb_accuracy_test)
	print "Accuracy Cos Test: " + str(cos_accuracy_test)

	print "Precision NB Train: " + str(nb_precision_train)
	print "Precision Cos Train: " + str(cos_precision_train)

	print "Precision NB Test: " + str(nb_precision_test)
	print "Precision Cos Test: " + str(cos_precision_test)

	print "Recall NB Train: " + str(nb_recall_train)
	print "Recall Cos Train: " + str(cos_recall_train)

	print "Recall NB Test: " + str(nb_recall_test)
	print "Recall Cos Test: " + str(cos_recall_test)

	# Compute ROC curve and ROC area for each class
	categories = [-1, 1]

	# Create figure
	fig = plt.figure()

	# Initialize dicts
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	svm_plt = fig.add_subplot(121)

	print svm_y_test_score.shape
	print nb_y_test_score.shape

	# Compute fpr, tpr, and auc for both categories
	fpr, tpr, _ = roc_curve(Ytest, svm_y_test_score)
	roc_auc = auc(fpr, tpr)

	# Plot the curves
	plt.plot(fpr, tpr, color='darkorange', lw=2,
	             label='ROC curve (area = {0:0.2f})'
	             ''.format(roc_auc))

	svm_plt.plot([0, 1], [0, 1], 'k--', lw=2)
	svm_plt.set_xlim([0.0, 1.0])
	svm_plt.set_ylim([0.0, 1.05])
	svm_plt.set_xlabel('False Positive Rate')
	svm_plt.set_ylabel('True Positive Rate')
	svm_plt.set_title('SVM with Cosine Similarity Kernel ROC Plot')
	svm_plt.legend(loc="lower right", fontsize=7)


	nb_plt = fig.add_subplot(122)

	# Initialize dicts
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	# Compute fpr, tpr, and auc for both categories
	for i, category in enumerate(categories):
		fpr[i], tpr[i], _ = roc_curve(Ytest, nb_y_test_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Plot the curves
	colors = cycle(['aqua', 'darkorange'])
	for i, color in zip(range(len(categories)), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=2,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))
	nb_plt.plot([0, 1], [0, 1], 'k--', lw=2)
	nb_plt.set_xlim([0.0, 1.0])
	nb_plt.set_ylim([0.0, 1.05])
	nb_plt.set_xlabel('False Positive Rate')
	nb_plt.set_ylabel('True Positive Rate')
	nb_plt.set_title('Naive Bayes ROC Plot')
	nb_plt.legend(loc="lower right", fontsize=7)

	# Save result as pdf
	plt.show()
