from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from xgboost import XGBRegressor


import numpy as np
import math
import pandas as pd
import numpy as np 
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output



 	
# class PlotLearning(keras.callbacks.Callback):
#     """
#     Callback to plot the learning curves of the model during training.
#     """
#     def on_train_begin(self, logs={}):
#         self.metrics = {}
#         for metric in logs:
#             self.metrics[metric] = []
            

#     def on_epoch_end(self, epoch, logs={}):
#         # Storing metrics
#         for metric in logs:
#             if metric in self.metrics:
#                 self.metrics[metric].append(logs.get(metric))
#             else:
#                 self.metrics[metric] = [logs.get(metric)]
        
#         # Plotting
#         metrics = [x for x in logs if 'val' not in x]
        
#         f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
#         clear_output(wait=False)

#         for i, metric in enumerate(metrics):
#             axs[i].plot(range(1, epoch + 2), 
#                         self.metrics[metric], 
#                         label=metric)
#             if logs['val_' + metric]:
#                 axs[i].plot(range(1, epoch + 2), 
#                             self.metrics['val_' + metric], 
#                             label='val_' + metric)
                
#             axs[i].legend()
#             axs[i].grid()

#         plt.tight_layout()
#         plt.show()



def apply_pca(X, standardize=True):
    # Standardize
    X = X.copy()
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings



def fill_mode(data, columns):
    data[columns] = data[columns].fillna(data[columns].mode().iloc[0])
    return data


def fill_median(data, columns):
    data[columns] = data[columns].fillna(data[columns].median())
    return data

def preprocess(data):
    

    data = fill_mode(data, ['VIP', 'HomePlanet', 'Destination', 
                            'CryoSleep'])
    data = fill_median(data, ['Age'])
    data['Name'] = data['Name'].fillna('No Name')
    data['Cabin'].ffill(inplace=True)
    data[['id1', 'id2']] = data.PassengerId.str.split("_", expand=True)
    data = data.drop(['PassengerId'], axis=1)

    data.Destination = data.Destination.str.split(expand=True)[0]
    data.Destination = data.Destination.str.split("-", expand=True)[0]

    data[['deck', 'num', 'side']] = data.Cabin.str.split("/", expand=True)
    data[['first', 'last']] = data.Name.str.split(expand=True)
    data = data.drop(['Cabin', 'Name', 'first'], axis=1)
    
    data[['id1', 'id2', 'num']] = data[['id1', 'id2', 'num']].apply(pd.to_numeric)
    data['last'], _ = data['last'].factorize()

    data_encoded = pd.get_dummies(data.select_dtypes("object"))
    data = data.drop(data.select_dtypes("object"), axis=1)    
    data = data.join(data_encoded)
    data.fillna(0, inplace=True)
    
    features_pca = [
        "id1", "last", "num", "CryoSleep"
    ]


    pca, X_pca, loadings = apply_pca(data.loc[:, features_pca])
    data = data.join(X_pca.PC2)




    
    # # Standardize

    # X_scaled = data.copy()
    # X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    # kmeans = KMeans(n_clusters=100, random_state=0)
    # data["Cluster"] = kmeans.fit_predict(X_scaled)
    return data

def nn_train_run(x_train, y_train, x_test, y_test):
    reg_val = 0.01
    model = tf.keras.models.Sequential([
  
  tf.keras.layers.Dense(28, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(43, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(27, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(13, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  
  tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
 
    ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(lr=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_split=0.1, epochs=100)
    p_train = model.predict(x_test)
    return history, model, p_train



train_data = pd.read_csv('train.csv')
y_train = train_data.Transported
train_data = train_data.drop(['Transported'], axis=1)
train_data = preprocess(train_data)
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)

(x_train, x_test, y_train, y_test) = train_test_split(train_data, y_train, test_size = .25)





history, model, p_train = nn_train_run(x_train, y_train, x_test, y_test)

plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(model.evaluate(x_test, y_test))





# test_data = pd.read_csv('test.csv')

# test_data_encoded = pd.get_dummies(test_data.select_dtypes("object"))
# test_data = test_data.drop(test_data.select_dtypes("object"), axis=1)

# test_data = test_data.join(test_data_encoded)

# test_data.fillna(0, inplace=True)
# scaler = MinMaxScaler()
# test_data = scaler.fit_transform(test_data)

# predictions = model.predict(test_data)

# final_pd = pd.DataFrame(test_data.Id, columns=['Id', 'SalePrice'])
# final_pd.SalePrice = predictions
# final_pd.to_csv('predictions_nn.csv', index=False)