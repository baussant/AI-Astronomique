import numpy as np # type: ignore
import pandas as pd # type: ignore
import warnings
import hvplot.pandas
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from IPython.display import display, HTML # Pour afficher les données dans Jupyter Notebook avec un format HTML
from tensorflow.keras.utils import to_categorical # type: ignore

def PCA(variance,X_scaled):
    # Réduction des dimensions
    pca = PCA(n_components=variance)  # Conserver XX% de la variance
    X_pca = pca.fit_transform(X_scaled)
    X_final = X_pca
    # Afficher ou utiliser le nombre de PCA pris en compte
    PCA_Number = pca._n_features_out
    return X_final,PCA_Number,pca

def Smote(neighbors_number,X_pca,Y,Random_State,encoder):
    # Rééquilibrage des classes avec SMOTE
    smote = SMOTE(random_state=Random_State,k_neighbors=neighbors_number)
    X_resampled, Y_resampled = smote.fit_resample(X_pca, Y)

    # Décodage des valeurs
    Y_resampled2 = pd.DataFrame()
    Y_resampled2['TypeCoreName'] = encoder.inverse_transform(Y_resampled['TypeCoreName'])
    print("----------------  Répartiton Donnée après Smote --------------\n")  
    typecore_count = Y_resampled2['TypeCoreName'].value_counts()
    typecore_table = typecore_count.to_frame().T  # Vision Horizontal
    print(typecore_table)
    print(f"\n\n")
    return X_resampled,Y_resampled

def Etiq_One_Hot(y,encoder):
    # Conversion des étiquettes en one-hot
    y_categorical = to_categorical(y, num_classes=len(encoder.classes_))        
    return y_categorical