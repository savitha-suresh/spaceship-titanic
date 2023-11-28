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
from xgboost import XGBRegressor, XGBClassifier


import numpy as np
import pandas as pd
import math


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


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


def xgboost_train_run(x_train, y_train, x_test, y_test):
    model = XGBClassifier(learning_rate=0.1, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0.1, subsample=0.7,
                                     colsample_bytree=0.7,
                                      nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
    model.fit(x_train, y_train)
    p_train = model.predict(x_test)
    return model, p_train

def fill_mode(data, columns):
    data[columns] = data[columns].fillna(data[columns].mode().iloc[0])
    return data


def fill_median(data, columns):
    data[columns] = data[columns].fillna(data[columns].median())
    return data

def preprocess(data):
    

    data = fill_mode(data, ['VIP', 'HomePlanet', 'Destination', 
                            'CryoSleep'])
    data = fill_median(data, ['Age', 'RoomService', 'VRDeck', 'FoodCourt', 'ShoppingMall', 'Spa'])
    data['Name'] = data['Name'].fillna('No Name')
    data['Cabin'].ffill(inplace=True)
    data[['id1', 'id2']] = data.PassengerId.str.split("_", expand=True)
    data = data.drop(['PassengerId'], axis=1)

    data.Destination = data.Destination.str.split(expand=True)[0]
    data.Destination = data.Destination.str.split("-", expand=True)[0]

    data[['deck', 'num', 'side']] = data.Cabin.str.split("/", expand=True)
    data[['first', 'last']] = data.Name.str.split(expand=True)
    data = data.drop([ "Cabin", 'first', 'Name'], axis=1)
    data['VIP'] = data['VIP'].astype(int)
    data['CryoSleep'] = data['CryoSleep'].astype(int)
    
    data[['id1', 'id2', 'num']] = data[['id1', 'id2', 'num']].apply(pd.to_numeric)
    data['last'], _ = data['last'].factorize()
    data_encoded = pd.get_dummies(data.select_dtypes("object"))
    data = data.drop(data.select_dtypes("object"), axis=1)    
    data = data.join(data_encoded)
    data.fillna(0, inplace=True)
    
    features_pca = [
        "id1", "last", "num", "CryoSleep", "Spa", "RoomService"
    ]


    pca, X_pca, loadings = apply_pca(data.loc[:, features_pca])
    data = data.join(X_pca.PC2)

    
    # Standardize
    # features = ['id2']
    
    
    # X_scaled = data.loc[:, features].copy()
    
    # X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    # kmeans = KMeans(n_clusters=8, random_state=0)
    # data["Cluster"] = kmeans.fit_predict(X_scaled)
    return data


train_data = pd.read_csv('train.csv')
y_train = train_data.Transported
train_data = train_data.drop(['Transported'], axis=1)
train_data = preprocess(train_data)

(x_train, x_test, y_train, y_test) = train_test_split(train_data, y_train, test_size = .25)

#model, p_train = nn_train_run(x_train, y_train, x_test, y_test)
model, p_train = xgboost_train_run(x_train, y_train, x_test, y_test)


print("Auc score - {}\n\n".format(roc_auc_score(y_test, p_train)))




test_data = pd.read_csv('test.csv')
final_pd = pd.DataFrame(test_data.PassengerId, columns=['PassengerId', 'Transported'])
test_data = preprocess(test_data)

predictions = model.predict(test_data)

final_pd.Transported = predictions
final_pd.Transported = final_pd.Transported.replace({1:'True', 0:'False'})
final_pd.to_csv('predictions_xgboost_latest.csv', index=False)

# score 0.15444 and score 0.15236 gave a rmse of 0.099