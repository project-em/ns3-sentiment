from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from dueling_lstms import reload_model
from dueling_lstms import label_sentence
from sentence_tool import fetch_sentences
from dueling_lstms import SourceStance

# Load models
cons_model = None
cons_vocab = None
lib_model = None
lib_vocab = None

# cons_model, cons_vocab = reload_model(SourceStance.conservative)
# lib_model, lib_vocab = reload_model(SourceStance.liberal)

# Load training and testing splits
sentences_training = open('data/training.dat', 'r')
sentences_testing = open('data/testing.dat', 'r')

Y_train = []
train_predictions = []


# Label each training sentence
for line in sentences_training.readlines():
    split = line.split("\t")
    Y_train.append(int(split[1]))
    sentence = split[0]
    label = label_sentence(cons_model, lib_model, cons_model, lib_vocab, sentence)
    train_predictions.append(label)

X_test = []
Y_test = []
test_predictions = []

# Label each testing sentence
for line in sentences_testing.readlines():
    split = line.split("\t")
    Y_test.append(int(split[1]))
    sentence = split[0]
    label = label_sentence(cons_model, lib_model, cons_model, lib_vocab, sentence)
    test_predictions.append(label)

# Calculate the training and testing precision scores
training_precision = precision_score(Y_train, train_predictions)
testing_precision = precision_score(Y_test, test_predictions)

print "Training precision: " + str(training_precision)
print "Testing precision: " + str(testing_precision)