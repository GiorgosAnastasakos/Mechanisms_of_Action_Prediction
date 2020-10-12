# HyperTune  - DimReduction

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA   
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics.classification import log_loss
from sklearn.utils import shuffle
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dropout, Dense
#from keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import EarlyStopping

import sys
sys.executable

train_features = pd.read_csv("~/Documents/R/Kaggle/lish_moa/train_features.csv")
train_targets_scored = pd.read_csv("~/Documents/R/Kaggle/lish_moa/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv("~/Documents/R/Kaggle/lish_moa/train_targets_nonscored.csv")

# join train_features and non_scored_forTrain
non_scored = pd.merge(train_targets_nonscored,train_features,on="sig_id", how="inner")   
score ={"score":non_scored.iloc[:,1:403].sum(axis=1)}
score=pd.DataFrame(data = score)
non_scored = pd.concat([non_scored, score], axis=1)
non_scored = non_scored[non_scored["score"]>0]
non_scored_forTrain = pd.concat([non_scored.iloc[:,0], 
                                 non_scored.iloc[:,403:1278]], axis=1)
train_features = pd.concat([train_features, non_scored_forTrain])
train_features_labels = pd.merge(train_features, train_targets_nonscored,
                                 how="left", on="sig_id")
train_features_labels = pd.merge(train_features, train_targets_scored, 
                                 how="left", on="sig_id")
# Features
train_features = train_features_labels.iloc[:, 1:876]
train_features.shape
# Labels
train_labels = train_features_labels.iloc[:, 876:1082]
train_labels.shape


def modelPca(n_components=80, n_estimators=50):        
    # PCA
    embedding = PCA(n_components=n_components)
    train_features_pca = train_features.iloc[:, 3:]
    train_pca = embedding.fit_transform(train_features_pca)
    train_pca.shape
    train_features["cp_time"] = train_features["cp_time"].astype(str)
    dummies_train = train_features.iloc[:, 0:3]
    dummies_train = pd.get_dummies(data=dummies_train, drop_first=True)
    train_features_pca_dummies = pd.concat([pd.DataFrame(data=train_pca),
                                            dummies_train], axis=1)
    test_size = 0.85
    X_train, X_test, y_train, y_test = train_test_split(
        train_features_pca_dummies.to_numpy(),
        train_labels.to_numpy(),
        test_size=test_size,
        random_state=42)
    # random forest multioutput
    xgb= XGBClassifier(n_estimators = n_estimators)
    global classifier
    classifier = MultiOutputClassifier(xgb)
    classifier.fit(X_train, y_train)
    val_preds = classifier.predict_proba(X_test)
    val_preds = np.array(val_preds)[:,:,1].T
    global loss
    loss = log_loss(np.ravel(y_test),np.ravel(val_preds))
    print("loss :",format(round(loss,5)))

#modelPca()

xgb_loss_df = pd.DataFrame()
n_components = [2,10,20,40,60,80,100]
n_estimators = [20,60,100,200,500]

for c in range(0,len(n_components),1):
    for e in range(0, len(n_estimators),1):
        
        print(c, " # n_components: ", n_components[c])
        print(e, " # n_estimators: ", n_estimators[e])
        
        modelPca(n_components = n_components[c], n_estimators = n_estimators[e])
        kouvas = pd.DataFrame(data = {'n_components':n_components[c],
                                      'n_estimators':n_estimators[e],
                                      'loss':round(loss,5)},index=[0]
                              )
        
        xgb_loss_df = pd.concat([xgb_loss_df, kouvas])
        xgb_loss_df.to_csv("xgb_loss_df.csv")


plt.plot(xgb_loss_df['n_estimators'], xgb_loss_df['loss'])
plt.show()
plt.plot(xgb_loss_df['n_components'], xgb_loss_df['loss'])

xgb_loss_df
'''
from sklearn.manifold import MDS

# Creating the function
def modelmds(n_components = 2, 
              test_size = 0.85,
              units = 256,
              optmz = "adamax",
              epochs = 100,
              batch = 64,
              lr_rate =0.1,
              patience = 5,
              Dropout_rate = 0.5,
              random_state = 42):
 
   #HyperTune umapDeapLearning
    
    embedding = MDS(n_components=n_components)
    
    features_for_lle = train_features.iloc[:,3:]
    train_lle = embedding.fit_transform(features_for_lle)
    train_lle.shape
    train_features["cp_time"]=train_features["cp_time"].astype(str)
    dummies_train = train_features.iloc[:,0:3]
    dummies_train = pd.get_dummies(data=dummies_train,drop_first=True)
    train_features_lle_dummies = pd.concat([pd.DataFrame(data=train_lle),dummies_train],axis=1)
    test_size = test_size
    X_train, X_test, y_train, y_test = train_test_split(train_features_lle_dummies.to_numpy(),
                                                        train_labels.to_numpy(),
                                                        test_size = test_size, 
                                                        random_state=42)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_rate,
                                  patience=patience)#, min_lr=0.001)
    callback_list = [reduce_lr]
    
    global model
    model = Sequential(
    [
    layers.Dense(units, activation = 'relu', input_shape=(train_features_lle_dummies.to_numpy().shape[1],)),
    layers.Dropout(rate = Dropout_rate),
    layers.Dense(units, activation = 'relu'), 
    layers.Dropout(rate = Dropout_rate),
    layers.Dense(units = train_labels.shape[1], activation = 'sigmoid')
    ]    
    )



    #model.summary()
    model.compile(
    optimizer = optmz,
    loss = "binary_crossentropy",
    metrics = ["accuracy"])

    history = model.fit(
        X_train, y_train,
        epochs = epochs,
        batch_size = batch,
        callbacks = callback_list,
        validation_data=(X_test,y_test),
        verbose=0
    )



    # list all data in history
    #print(history.history.keys())
    global valLoss
    valLoss = round(np.min(history.history["val_loss"]),6)
    print("best val_loss:", round(np.min(history.history["val_loss"]),6))
    print("best loss:", round(np.min(history.history["loss"]),6))

    plt.figure(figsize = (10,6))
    plt.rc_context({'xtick.color':'white', 'ytick.color':'white'})
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(figsize = (10,6))
    plt.rc_context({'xtick.color':'white', 'ytick.color':'white'})
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

modelmds()

'''
     