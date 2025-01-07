import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,roc_curve,auc
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter
from ToolsAI import Calcul_XX_YY
from ModelHistory import save_knn_results
from Graphique import Affichage_accuracy, Affichage_confusion_matrix, Affichage_roc_curve,Affichage_courbe_apprentissage,Affichage_proba2,Affichage_base
from ToolsAI import split_data, apply_smote, apply_pca,sampling
from EcritureCSV import create_csv

def KNN_Classifier(smote_status,sampling_number,sampling_strategy,data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):
   
    type_planet_df = data.copy()
    
    # Filtrer les classes avec suffisamment de données
    class_counts = type_planet_df[Y_Target].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    type_planet_df = type_planet_df[type_planet_df[Y_Target].isin(valid_classes)]

    if type_planet_df.empty:
        raise ValueError("Aucune classe valide n'a suffisamment de données pour l'analyse.")
    
    # Utilisation de la fonction Calcul_XX_YY
    X, Y, encoder = Calcul_XX_YY(type_planet_df,Y_Target,X_Chara)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Étape 1 : Diviser les données
    #X_train, X_test, Y_train, Y_test = split_data(X, Y) 
    X_train, X_test, Y_train, Y_test = split_data(X_scaled, Y) 
    # Etape 2 : Appliquer SMOTE
    if smote_status == True:
        if X_train.shape[0] < 10:
            raise ValueError("Pas assez de données pour appliquer SMOTE.")
        # Vérification des correspondances entre sampling_strategy et les valeurs encodées de Y
        sampling_strategy2,sampling_number = sampling(sampling_strategy,Y_train,encoder,sampling_number)
        # Étape 2 : Appliquer SMOTE
        X_train_resampled, Y_train_resampled = apply_smote(X_train, Y_train, valid_classes, sampling_strategy2,sampling_number)
        # Décodage pour affichage
        Y_resampled_decoded = pd.DataFrame({'TypeCoreName': encoder.inverse_transform(Y_train_resampled)})
        print("----------------  Répartition Données après SMOTE --------------\n")
        typecore_count = Y_resampled_decoded['TypeCoreName'].value_counts()
        typecore_table = typecore_count.to_frame().T
        print(typecore_table)
    else:
        X_train_resampled, Y_train_resampled = X_train, Y_train    


    X_train_scaled=X_train_resampled
    # Étape 4 : Réduction des dimensions (PCA)
    if PCA_State:
        X_pca_train, X_pca_test, PCA_Number,pca = apply_pca(X_train_scaled, X_test, variance)
    else:
        X_pca_train, X_pca_test = X_train_scaled, X_test
        PCA_Number = len(X.columns)

        # Vérification après PCA
    if X_pca_train.shape[0] < 10:
        raise ValueError("Pas assez de données après réduction de dimensionnalité pour poursuivre.")

     # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {'n_neighbors': range(1, neighbor_max), 'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan', 'minkowski']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=Value_cv,scoring='accuracy')
    grid_search.fit(X_pca_train, Y_train_resampled)

    # Meilleur paramètre k
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    # Meilleur modèle
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Affichage des informations de base
    Affichage_base(X_pca_train, X_pca_test, encoder, PCA_Number, best_k, grid_search)

    # Entraînement du modèle k-NN avec le meilleur k
    best_model = grid_search.best_estimator_
    best_model.fit(X_pca_train, Y_train_resampled)

    # Prédictions test et train
    Y_test_pred = best_model.predict(X_pca_test)
    Y_train_pred = best_model.predict(X_pca_train)
    Y_test_proba = best_model.predict_proba(X_pca_test)

    # Précision sur l'ensemble d'entraînement
    report,report_train,accuracy,train_accuracy=Affichage_accuracy(Y_test, Y_test_pred, Y_train_resampled, Y_train_pred, encoder)
   
   # Affichage des graphiques
    fig_confusion = Affichage_confusion_matrix(Y_test, Y_test_pred, encoder)
    fig_learning_curve = Affichage_courbe_apprentissage(best_model, X_pca_train, Y_train_resampled,Value_cv)
    fig_roc = Affichage_roc_curve(Y_test, Y_test_pred, encoder, best_model, X_pca_test)
    fig_proba = Affichage_proba2(Y_test_proba,valid_classes)

    if Save_Model:
        Smote = smote_status
        # Définir les graphiques à sauvegarder
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve",  fig_learning_curve))
        images.append(("Probability Distributions by Class", fig_proba))  
        images.append(("ROC Curve", fig_roc))  

        save_knn_results(sampling_number,sampling_strategy,Smote,PCA_Number,best_params,best_model,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)


