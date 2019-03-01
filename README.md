# Multi-Classification Neural Network Project

We can immediately recognize this as a  multi-classification problem {A,B,C,D,E}.
The steps we take in code are listed below:


### Recognize Dependencies:

`keras` : use for our Multi-class building

`scikit-learn` & `pandas`: preprocessing



### 1) Initial Data Analysis
To find the features of that are of interest, we can run a Recursive Feature Elimination.
Other options that could have been implemented to reduce the dimension of this set is PCA or Principal component Analysis).

### 2)  Fit some ML model(s).
After our data is processed, we can use a multi-classification Neural Network.
The depth is up to the designer but we can use several activation functions and the KerasClassifier. (KerasClassifier)

### 3) X-Validation
We can use 10 fold `StratifiedKFold` to validate our model. The cross-validation allows us to assess the performance of the model.

### 4) Next Steps
a) Experiment with hyper-parameters on the model -> could add more layers
b) Readjust our feature selection (could use PCA, or another strategy to detect significant features)
c) Change the count of features we selected (could use more or less than 10)
d) Increase Validation:
e) Increase Data-set