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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from IPython.display import display, HTML # Pour afficher les données dans Jupyter Notebook avec un format HTML
from tensorflow.keras.utils import to_categorical # type: ignore
from collections import Counter

def split_data(X, Y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

def apply_smote(X_train, Y_train, valid_classes, sampling_strategy):

    smote_neighbors = min(3, max(1, len(valid_classes) - 1))
    smote = SMOTE(random_state=42, k_neighbors=smote_neighbors, sampling_strategy=sampling_strategy)
    X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

    return X_resampled, Y_resampled  

def apply_BorderlineSMOTE(X_train, Y_train, valid_classes, sampling_strategy):

    smote_neighbors = min(3, max(1, len(valid_classes) - 1))
    borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=smote_neighbors)
    X_resampled, Y_resampled = borderline_smote.fit_resample(X_train, Y_train)

    return X_resampled, Y_resampled

def apply_ADASYN(X_train, Y_train, valid_classes, sampling_strategy):

    smote_neighbors = min(3, max(1, len(valid_classes) - 1))
    adasyn = ADASYN(random_state=42, n_neighbors=smote_neighbors)
    X_resampled, Y_resampled = adasyn.fit_resample(X_train, Y_train)

    return X_resampled, Y_resampled

def apply_pca(X_resampled, X_test, variance):
    """
    Réduit les dimensions avec PCA sur les ensembles d'entraînement et de test.
    """
    pca = PCA(n_components=variance)
    X_pca_train = pca.fit_transform(X_resampled)
    X_pca_test = pca.transform(X_test)
    print(f"Nombre de composantes principales retenues : {pca.n_components_}")
    return X_pca_train, X_pca_test, pca.n_components_,pca        

def sampling(sampling_strategy,Y,encoder,sampling_number):
    if isinstance(sampling_strategy, dict):       # Si Y est encodé, ajuster sampling_strategy aux indices
        decoded_classes = encoder.inverse_transform(np.unique(Y))  # Classes originales
        sampling_strategy2 = {
            encoder.transform([cls])[0]: sampling_strategy[cls]
            for cls in sampling_strategy
            if cls in decoded_classes
        }
        if not sampling_strategy2:
            raise ValueError("Aucune des classes spécifiées dans 'sampling_strategy' n'est présente dans les données après encodage.")
    elif sampling_strategy == True:
        # Stratégie par défaut pour équilibrer toutes les classes
        # Assurez-vous que Y est une liste unidimensionnelle de classes encodées
        if isinstance(Y, pd.DataFrame):
            YY = Y.iloc[:, 0].tolist()  # Convertir en liste si c'est un DataFrame avec une seule colonne
        elif isinstance(Y, pd.Series):
            YY = Y.tolist()  # Convertir en liste si c'est une Series
        class_counts2 = Counter(YY)
        max_target = sampling_number
        sampling_strategy2 = {
        cls: max(count, max_target)  # Garder la fréquence actuelle si elle est supérieure à target_samples
        for cls, count in class_counts2.items()
        }
    else:
        sampling_number = 'N/A'
        sampling_strategy2 = 'auto'  
    return sampling_strategy2,sampling_number   

def Calcul_XX_YY(data,Y_Target,X_Chara):
    
    XR = data.copy()

    XR.columns = XR.columns.str.strip()

    # Vérification des colonnes spécifiées
    if Y_Target not in XR.columns:
        raise ValueError(f"La colonne spécifiée pour Y_Target ('{Y_Target}') est absente des données.")
    
      # Encodage de la colonne cible si classification
    encoder = LabelEncoder()
    XR[Y_Target] = encoder.fit_transform(XR[Y_Target].str.strip())
    print(f"Classes encodées : {list(encoder.classes_)}")

    XX = XR.drop(columns=[Y_Target])
    XX = XX.select_dtypes(include=['number']).dropna(axis=1)  # Suppression des colonnes non numériques
    XX = XX[X_Chara]

    YY = XR[[Y_Target]].dropna()

    return XX,YY,encoder

def Calcul_XX_YY_2(data,Y_Target,X_Chara):
    
    XR = data.copy()

    XR.columns = XR.columns.str.strip()

    # Vérification des colonnes spécifiées
    if Y_Target not in XR.columns:
        raise ValueError(f"La colonne spécifiée pour Y_Target ('{Y_Target}') est absente des données.")

    # Encodage de la colonne cible pour la classification
    encoder = LabelEncoder()
    XR[Y_Target] = encoder.fit_transform(XR[Y_Target].str.strip())
    print(f"Classes encodées : {list(encoder.classes_)}")

    # Séparation des caractéristiques et de la cible
    XX = XR[X_Chara].dropna()  # Assurez-vous que seules les colonnes numériques sélectionnées sont utilisées
    YY = XR.loc[XX.index, Y_Target]  # Garder les mêmes indices pour Y

    return XX, YY, encoder

