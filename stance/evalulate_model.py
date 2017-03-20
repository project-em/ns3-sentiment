from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import load_model
from dueling_lstms import label_sentence
from sentence_tool import fetch_sentences

# Load models
conservative_model = load_model('conservative_model.h5')

liberal_model = load_model('liberal_model.h5')

sentences = fetch_sentences()

# How to get initial labels of liberal/conservative in the correct order (currently they are stored in 2 different files)
