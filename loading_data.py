from utilities import load_data
from utilities import extract_feature
from utilities import int2emotion, AVAILABLE_EMOTIONS


X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = load_data(test_size=0.25,extractor="mfcc")
X_train_chroma, X_test_chroma, y_train_chroma, y_test_chroma = load_data(test_size=0.25,extractor="chroma")
X_train_tonnetz, X_test_tonnetz, y_train_tonnetz, y_test_tonnetz = load_data(test_size=0.25,extractor="tonnetz")
#X_train_mel, X_test_mel, y_train_mel, y_test_mel = load_data(test_size=0.25,extractor="mel")

#print(X_test, y_test)
# print some details
# number of samples in training data
#print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
#print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted
# using utils.extract_features() method
#no_features = X_train.shape[1]
#print("[+] Number of features:", X_train.shape[1])
