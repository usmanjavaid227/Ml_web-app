# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App ki hadding
st.write("""
# Explore different Ml models and datasets
Daikhte han kon sa best ha in main se !
""")

# define datasets name in sidebar
datasets_names = st.sidebar.selectbox('Select Dataset',('Iris','Breast Cancer','Wine','Digits'))
classifier_names = st.sidebar.selectbox('Select Classifier',('KNN','SVM','Random Forest'))

# function to load Datasets
def get_dataset(datasets_names):
    data =  None 
    if datasets_names == 'Iris':
        data = datasets.load_iris()
    elif datasets_names == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    elif datasets_names == 'Wine':
        data = datasets.load_wine()
    elif datasets_names == 'Digits':
        data = datasets.load_digits()
    else:
        st.write('Invalid Dataset')
    x=data.data
    y=data.target
    return x,y

# call function
X,y = get_dataset(datasets_names)

st.write('Dataset Shape: ',X.shape)
st.write('No of classes: ',len(np.unique(y)))

# add different classifier parameter in user inputs
def add_parameter_ui(classifier_names):
    params=dict() # create empty dictionary
    if classifier_names == 'KNN':
        K = st.sidebar.slider('Select K',1,15)
        params['K'] = K   # no of neighbours
    elif classifier_names == 'SVM':
        C = st.sidebar.slider('Select C',0.01,10.0)
        params['C'] = C  # it's the degree of correct classification
    else:
        max_depth = st.sidebar.slider('Select Max Depth',1,15)
        params['max_depth'] = max_depth # max depth of the tree
        n_estimators = st.sidebar.slider('Select n_estimators',1,100)
        params['n_estimators'] = n_estimators
    return params
#####
#####
# call function
params = add_parameter_ui(classifier_names)
        
#####
#####
def get_classifier(classifier_names,params):
    clf = None
    if classifier_names == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_names == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    return clf

# call function
clf = get_classifier(classifier_names,params)

# split datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

# train classifier
clf.fit(X_train,y_train)
# predict classifier
y_pred = clf.predict(X_test)
# calculate accuracy
acc = accuracy_score(y_test,y_pred)
st.write(f'classifer name ={classifier_names}')
st.write('Accuracy: ',acc)

# adding plotting
pca=PCA(2)
X_projected = pca.fit_transform(X)
# sciling data
x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('PCA 1',fontsize=15)
plt.ylabel('PCA 2',fontsize=15)
plt.title('PCA',fontsize=20)
st.pyplot(fig)

