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
from RandomForest import Calcul_XX_YY
from ModelHistory import save_knn_results
from Graphique import Affichage_accuracy, Affichage_confusion_matrix, Affichage_roc_curve,Affichage_courbe_apprentissage,Affichage_proba
from EcritureCSV import create_csv

def KNN_Classifier(data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):

    type_planet_df = data.copy()

    # Filtrer les classes avec suffisamment de données
    class_counts = type_planet_df['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    type_planet_df = type_planet_df[type_planet_df['TypeCoreName'].isin(valid_classes)]
    
    X, Y, encoder = Calcul_XX_YY(type_planet_df, Y_Target, X_Chara)

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Étape 1 : Diviser les données
    X_train, X_test, Y_train, Y_test = split_data(X_scaled, Y)

    # Réduction des dimensions (PCA)
    if PCA_State:
        X_pca_train, X_pca_test, PCA_Number,pca = apply_pca(X_train, X_test, variance)
    else:
        X_pca_train, X_pca_test = X_train, X_test
        PCA_Number = len(X.columns)

        # Vérification après PCA
    if X_pca_train.shape[0] < 10:
        raise ValueError("Pas assez de données après réduction de dimensionnalité pour poursuivre.")
    """
    if PCA_State == True:
        # Réduction des dimensions
        pca = PCA(n_components=variance)  # Conserver XX% de la variance
        X_pca = pca.fit_transform(X_scaled)
        X_final = X_pca
        # Afficher ou utiliser le nombre de PCA pris en compte
        PCA_Number = pca._n_features_out
        print(f"Nombre de composantes principales retenues : {pca.n_components_}")
    else:
        X_final = X_scaled
        PCA_Number = len(X.columns)
        

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X_final, Y.reset_index(drop=True), test_size=0.2, random_state=42)
    """

    # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {
        'n_neighbors': range(1, neighbor_max),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=Value_cv, scoring='accuracy')
    grid_search.fit(X_pca_train, Y_train)

    # Meilleur paramètre k 
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Affichage des informations de base
    Affichage_base(X_pca_train, X_pca_test, encoder, PCA_Number, best_k, grid_search)
  
    # Entraînement du modèle k-NN avec le meilleur k
    knn = grid_search.best_estimator_
    knn.fit(X_pca_train, Y_train)
        
    # Prédictions
    y_pred = knn.predict(X_pca_test)

    # Précision sur l'ensemble d'entraînement
    y_train_pred = knn.predict(X_pca_train)
   
    # Évaluation des performances
    report,report_train,accuracy,train_accuracy=Affichage_accuracy(Y_test, y_pred, Y_train, y_train_pred, encoder)

    # Affichage des graphiques
    fig_confusion,fig_CurveLearning,fig_Predict,fig_roc = Affichage_graphique(Y_test, y_pred, encoder, X_pca_train, Y_train, knn, Value_cv, valid_classes,X_pca_test)
    """
    report,report_train,accuracy,train_accuracy =  Affichage_accuracy(Y_test, Y_test_pred,Y_train_resampled, Y_train_pred,encoder)
    fig_confusion = Affichage_confusion_matrix(Y_test, Y_test_pred, encoder)
    fig_learning_curve = Affichage_courbe_apprentissage(best_model, X_train_resampled, Y_train_resampled, cv_data)
    fig_roc = Affichage_roc_curve(Y_test, Y_test_pred, encoder, best_model, X_test)
    fig_proba = Affichage_proba(y_test_proba)
    """

    # Affichage des résultats
    report,report_train,images,accuracy,train_accuracy = Affichage(Y_test, Y_test_pred, classification, encoder, Y_train_resempled, Y_train_pred,best_model,X_train_resempled,X_test,cv_data,valid_classes)  # Retour Report, image,accuracy,train_accuracy


    if Save_Model:
        # Définir les graphiques à sauvegarder
        Smote = False
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve", fig_CurveLearning))
        images.append(("Probability Distributions by Class", fig_Predict))  
        images.append(("ROC Curve", fig_roc))  
        """
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("ROC Curve",  fig_roc))
        images.append(("Learning Curve", fig_learning_curve))
        images.append(("Probability", fig_proba))
        """
        sampling_number = 'N/A'
        sampling_strategy = False
        save_knn_results(sampling_number,sampling_strategy,Smote,PCA_Number,best_params,knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)


def KNN_Classifier_Smote2(sampling_number,sampling_strategy,data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):
   
    type_planet_df = data.copy()

    # Filtrer les classes avec suffisamment de données
    class_counts = type_planet_df[Y_Target].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    type_planet_df = type_planet_df[type_planet_df[Y_Target].isin(valid_classes)]

    if type_planet_df.empty:
        raise ValueError("Aucune classe valide n'a suffisamment de données pour l'analyse.")
    
    # Utilisation de la fonction Calcul_XX_YY
    X, Y, encoder = Calcul_XX_YY(type_planet_df,Y_Target,X_Chara)

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if X_scaled.shape[0] < 10:
        raise ValueError("Pas assez de données pour appliquer SMOTE.")

    # Vérification des correspondances entre sampling_strategy et les valeurs encodées de Y
    sampling_strategy2,sampling_number = sampling(sampling_strategy,Y,encoder,sampling_number)

    # Étape 1 : Diviser les données
    X_train, X_test, Y_train, Y_test = split_data(X_scaled, Y)

    # Étape 2 : Appliquer SMOTE
    X_resampled, Y_resampled = apply_smote(X_train, Y_train, valid_classes, sampling_strategy2,sampling_number)

    # Décodage pour affichage
    Y_resampled_decoded = pd.DataFrame({'TypeCoreName': encoder.inverse_transform(Y_resampled)})
    print("----------------  Répartition Données après SMOTE --------------\n")
    typecore_count = Y_resampled_decoded['TypeCoreName'].value_counts()
    typecore_table = typecore_count.to_frame().T
    print(typecore_table)

    # Étape 3 : Réduction des dimensions (PCA)
    if PCA_State:
        X_pca_train, X_pca_test, PCA_Number,pca = apply_pca(X_resampled, X_test, variance)
    else:
        X_pca_train, X_pca_test = X_resampled, X_test
        PCA_Number = len(X.columns)

        # Vérification après PCA
    if X_pca_train.shape[0] < 10:
        raise ValueError("Pas assez de données après réduction de dimensionnalité pour poursuivre.")

     # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {'n_neighbors': range(1, neighbor_max), 'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan', 'minkowski']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=Value_cv,scoring='accuracy')
    grid_search.fit(X_pca_train, Y_resampled)

    # Meilleur paramètre k
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    # Meilleur modèle
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Affichage des informations de base
    Affichage_base(X_pca_train, X_pca_test, encoder, PCA_Number, best_k, grid_search)

    # Entraînement du modèle k-NN avec le meilleur k
    knn = grid_search.best_estimator_
    knn.fit(X_pca_train, Y_resampled)

    # Prédictions test et train
    Y_pred = knn.predict(X_pca_test)
    Y_train_pred = knn.predict(X_pca_train)

    # Précision sur l'ensemble d'entraînement
    report,report_train,accuracy,train_accuracy=Affichage_accuracy(Y_test, Y_pred, Y_resampled, Y_train_pred, encoder)
  
    # Affichage des graphiques
    fig_confusion,fig_CurveLearning,fig_Predict,fig_roc = Affichage_graphique(Y_test, Y_pred, encoder, X_pca_train, Y_resampled, knn, Value_cv, valid_classes,X_pca_test)
  
    if Save_Model:
        Smote = True
        # Définir les graphiques à sauvegarder
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve", fig_CurveLearning))
        images.append(("Probability Distributions by Class", fig_Predict))  
        images.append(("ROC Curve", fig_roc))  

        save_knn_results(sampling_number,sampling_strategy,Smote,PCA_Number,best_params,knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)

def Affichage_base(x_train, x_test, encoder, PCA_Number, best_k, grid_search):
    print(f"-----------------------------------------------------------------------------------------\n")
    print(f"Entraînement sur {x_train.shape[0]} échantillons, test sur {x_test.shape[0]} échantillons")
    print(f"Nombre de classes : {len(encoder.classes_)}")
    print(f"Classes : {encoder.classes_}")
    print(f"Nombre de composantes principales : {PCA_Number}")
    print(f"Nombre de voisins : {best_k}")
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")
    print(f"-----------------------------------------------------------------------------------------\n")

def Affichage_accuracy(y_test, y_pred, y_train, y_train_pred, encoder):
    # Évaluation des performances
    valid_labels = encoder.transform(encoder.classes_)  # Classes valides dans l'ordre des encodages
    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    report = classification_report(y_test, y_pred, labels=valid_labels, target_names=encoder.classes_)
    report_train = classification_report(y_train, y_train_pred, labels=valid_labels, target_names=encoder.classes_)
    print(f"-----------------------------------------------------------------------------------------\n")
    print(f"--------------------- Ecart bias-variance ----------------------------------\n")
    bias_variance = accuracy - train_accuracy
    print(f"Ecart bias-variance: {bias_variance:.2f}\n")
    print(f"-----------------------------------------------------------------------------------------\n")
    if bias_variance < -0.1:
        print(f"Surapprentissage.")
        print(f"-----------------------------------------------------------------------------------------\n")
    elif bias_variance > 0.1:
        print(f"Sous-apprentissage.")
        print(f"-----------------------------------------------------------------------------------------\n")    
    else:
        print(f"Modèle équilibré.")
        print(f"-----------------------------------------------------------------------------------------\n")   

    # Affichage des résultats
    print(f"--------------------- Accuracy ----------------------------------\n")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"-----------------------------------------------------------------------------------------\n")
    print(f"--------------------- Classification Report Test ----------------------------------\n")
    print(report)
    print(f"-----------------------------------------------------------------------------------------\n")
    print(f"--------------------- Classification Report Train ----------------------------------\n")
    print(report_train)    
    return report,report_train,accuracy,train_accuracy

def Affichage_graphique(y_test, y_pred, encoder, X_final, Y, knn, cv_value, valid_classes,x_test):

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables')
    plt.title('Matrice de confusion')
    fig_confusion = plt.gcf()
    plt.show()

    # Courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(knn, X_final, Y, cv=cv_value, scoring='accuracy')
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    fig_CurveLearning = plt.gcf()
    plt.show()

    # Distribution des probabilités de Prédiction
    pred_probs = knn.predict_proba(x_test)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(valid_classes):
        sns.kdeplot(pred_probs[:, i], label=f'Class {class_name}')
    plt.title('Probability Distributions by Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    fig_Predict = plt.gcf()
    plt.show()
 
    # Binariser les classes
    # Binarisation des étiquettes
    y_test_binarized = label_binarize(y_test, classes=range(len(encoder.classes_)))
    n_classes = y_test_binarized.shape[1]

    # Courbe ROC
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonale
    # Calcul des courbes ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Classe {encoder.classes_[i]} (AUC = {roc_auc:.2f})")
    plt.title("Courbes ROC Multiclasses - KNN")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    fig_roc = plt.gcf()
    plt.show()

    return fig_confusion,fig_CurveLearning,fig_Predict,fig_roc


def split_data(X, Y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def apply_smote(X_train, Y_train, valid_classes, sampling_strategy, sampling_number, random_state=42):
    """
    Applique SMOTE pour rééquilibrer les classes dans l'ensemble d'entraînement.
    """
    smote_neighbors = min(3, max(1, len(valid_classes) - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=smote_neighbors, sampling_strategy=sampling_strategy)
    X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

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
        max_samples = max(class_counts2.values())
        #sampling_strategy2 = {cls: max_samples for cls in class_counts2.keys()}
        max_target = sampling_number
        sampling_strategy2 = {
        cls: max(count, max_target)  # Garder la fréquence actuelle si elle est supérieure à target_samples
        for cls, count in class_counts2.items()
        }
    else:
        sampling_number = 'N/A'
        sampling_strategy2 = 'auto'  
    return sampling_strategy2,sampling_number   

   