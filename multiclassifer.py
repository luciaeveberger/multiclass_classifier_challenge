from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def create_one_hot_encoded(Y):
    """ allows us to compare the categorical values"""
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    return encoded_Y


def select_features(X, Y, feature_selection):
    model = LogisticRegression()
    rfe = RFE(model, feature_selection)
    fit = rfe.fit(X, Y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


seed = 7
numpy.random.seed(seed)

dataset = pd.read_csv("data/sample.csv").values
# split into input (X) and output (Y) variables
X = dataset[:, 0:295]
Y = dataset[:, 295]

select_features(X, Y, feature_selection=10)
# encode the y features
y_encoded = create_one_hot_encoded(Y)
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y_encoded, cv=kfold)
print(results.mean())