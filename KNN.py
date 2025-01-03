import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from RandomForest import Calcul_XX_YY
from ModelHistory import save_knn_results

def KNN_Classifier(data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):

    """
    Fonction pour traiter un type de planète, effectuer l'entraînement K-NN et afficher les résultats.
    
    Parameters:
    - data: Le DataFrame contenant toutes les planètes avec leurs caractéristiques.
    - Echantillon_min: Nombre minimal de données pour traiter un type de planète.
    - Value_cv: Nombre de validations croisées pour GridSearch.
    - variance: Pourcentage de variance expliquée à conserver lors de la réduction dimensionnelle.
    - PCA_State: Booléen pour activer ou désactiver PCA.
    """
    
    type_planet_df = data.copy()

    # Filtrer les classes avec suffisamment de données
    class_counts = type_planet_df['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    type_planet_df = type_planet_df[type_planet_df['TypeCoreName'].isin(valid_classes)]
    
    X, Y, encoder = Calcul_XX_YY(type_planet_df, Y_Target, X_Chara)

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
        
    # Validation croisée K-Fold
    kf = KFold(n_splits=Value_cv, shuffle=True, random_state=42)
    print(f"Validation croisée K-Fold avec {Value_cv} splits")
    
    # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {
        'n_neighbors': range(1, neighbor_max),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kf)
    grid_search.fit(X_final, Y)

    # Meilleur paramètre k 
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X_final, Y.reset_index(drop=True), test_size=0.2, random_state=42)
 
    # Entraînement du modèle k-NN avec le meilleur k
    knn = grid_search.best_estimator_
    knn.fit(x_train, y_train)
        
    # Prédictions
    y_pred = knn.predict(x_test)

    # Précision sur l'ensemble d'entraînement
    y_train_pred = knn.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Évaluation des performances
    valid_labels = encoder.transform(encoder.classes_)  # Classes valides dans l'ordre des encodages
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=valid_labels, target_names=encoder.classes_)

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print("\nClassification Report:\n")
    print(report)
 
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
    train_sizes, train_scores, test_scores = learning_curve(knn, X_final, Y, cv=kf, scoring='accuracy')
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

    if Save_Model:
        # Définir les graphiques à sauvegarder
        Smote = False
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve", fig_CurveLearning))
        images.append(("Probability Distributions by Class", fig_Predict))  

        save_knn_results(Smote,PCA_Number,best_params,knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)


def KNN_Classifier_Smote(data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):
    """
    Fonction pour traiter un type de planète, effectuer l'entraînement K-NN et afficher les résultats.

    Parameters:
    - data: Le DataFrame contenant toutes les planètes avec leurs caractéristiques.
    - Echantillon_min: Nombre minimal de données pour traiter un type de planète.
    - Value_cv: Nombre de validations croisées pour GridSearch.
    - variance: Pourcentage de variance expliquée à conserver lors de la réduction dimensionnelle.
    - X_Chara: Liste des caractéristiques explicatives (features).
    - Y_Target: Nom de la colonne cible (target).
    - PCA_State: Booléen indiquant si PCA est utilisé ou non.
    - neighbor_max: Nombre maximum de voisins pour la recherche de k optimal.
    - Save_Model: Booléen indiquant si le modèle et les résultats doivent être sauvegardés.
    """
    
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

    # Réduction des dimensions (PCA)
    if PCA_State:
        pca = PCA(n_components=variance)
        X_pca = pca.fit_transform(X_scaled)
        print(f"Nombre de composantes principales retenues : {pca.n_components_}")
        PCA_Number = pca._n_features_out
    else:
        X_pca = X_scaled
        PCA_Number = len(X.columns)

        # Vérification après PCA
    if X_pca.shape[0] < 10:
        raise ValueError("Pas assez de données après réduction de dimensionnalité pour poursuivre.")
    
     # Rééquilibrage des classes avec SMOTE
    smote = SMOTE(random_state=42,k_neighbors=3)
    X_resampled, Y_resampled = smote.fit_resample(X_pca, Y)

    # Décodage des valeurs
    Y_resampled2 = pd.DataFrame()
    Y_resampled2['TypeCoreName'] = encoder.inverse_transform(Y_resampled['TypeCoreName'])
    print("----------------  Répartiton Donnée après Smote --------------\n")  
    typecore_count = Y_resampled2['TypeCoreName'].value_counts()
    typecore_table = typecore_count.to_frame().T  # Vision Horizontal
    print(typecore_table)
    print(f"\n\n")

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, random_state=42)

    # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {'n_neighbors': range(1, neighbor_max), 'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan', 'minkowski']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=Value_cv,scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    # Meilleur paramètre k
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    # Meilleur modèle
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Entraînement du modèle k-NN avec le meilleur k
    knn = grid_search.best_estimator_
    knn.fit(X_train, Y_train)

    # Prédictions
    Y_pred = knn.predict(X_test)

    # Précision sur l'ensemble d'entraînement
    y_train_pred = knn.predict(X_train)
    train_accuracy = accuracy_score(Y_train, y_train_pred)

    # Évaluation des performances
    valid_labels = encoder.transform(encoder.classes_)  # Classes valides dans l'ordre des encodages
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, labels=valid_labels, target_names=encoder.classes_)

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print("\nClassification Report:\n")
    print(report)
    print("\n")
    print(classification_report(Y_test, Y_pred))

    # Matrix de confusion
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig_confusion = plt.gcf()
    plt.show()

    # Courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(knn, X_pca, Y, cv=Value_cv, scoring='accuracy')
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
    pred_probs = knn.predict_proba(X_test)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(valid_classes):
        sns.kdeplot(pred_probs[:, i], label=f'Class {class_name}')
    plt.title('Probability Distributions by Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    fig_Predict = plt.gcf()
    plt.show()

    if Save_Model:
        Smote = True
        # Définir les graphiques à sauvegarder
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve", fig_CurveLearning))
        images.append(("Probability Distributions by Class", fig_Predict))  

        save_knn_results(Smote,PCA_Number,best_params,knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)

def KNN_Classifier_Smote2(data, Echantillon_min, Value_cv, variance, X_Chara, Y_Target, PCA_State,neighbor_max,Save_Model):
    """
    Fonction pour traiter un type de planète, effectuer l'entraînement K-NN et afficher les résultats.

    Parameters:
    - data: Le DataFrame contenant toutes les planètes avec leurs caractéristiques.
    - Echantillon_min: Nombre minimal de données pour traiter un type de planète.
    - Value_cv: Nombre de validations croisées pour GridSearch.
    - variance: Pourcentage de variance expliquée à conserver lors de la réduction dimensionnelle.
    - X_Chara: Liste des caractéristiques explicatives (features).
    - Y_Target: Nom de la colonne cible (target).
    - PCA_State: Booléen indiquant si PCA est utilisé ou non.
    - neighbor_max: Nombre maximum de voisins pour la recherche de k optimal.
    - Save_Model: Booléen indiquant si le modèle et les résultats doivent être sauvegardés.
    """
    
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

         # Rééquilibrage des classes avec SMOTE
    smote = SMOTE(random_state=42,k_neighbors=3)
    X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

        # Décodage des valeurs
    Y_resampled2 = pd.DataFrame()
    Y_resampled2['TypeCoreName'] = encoder.inverse_transform(Y_resampled['TypeCoreName'])
    print("----------------  Répartiton Donnée après Smote --------------\n")  
    typecore_count = Y_resampled2['TypeCoreName'].value_counts()
    typecore_table = typecore_count.to_frame().T  # Vision Horizontal
    print(typecore_table)
    print(f"\n\n")

    # Réduction des dimensions (PCA)
    if PCA_State:
        pca = PCA(n_components=variance)
        X_pca = pca.fit_transform(X_resampled)
        print(f"Nombre de composantes principales retenues : {pca.n_components_}")
        PCA_Number = pca._n_features_out
    else:
        X_pca =  X_resampled
        PCA_Number = len(X.columns)

        # Vérification après PCA
    if X_pca.shape[0] < 10:
        raise ValueError("Pas assez de données après réduction de dimensionnalité pour poursuivre.")
  
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(
    X_pca, Y_resampled, test_size=0.2, random_state=42)

    # Recherche des meilleurs hyperparamètres pour k (nombre de voisins)
    param_grid = {'n_neighbors': range(1, neighbor_max), 'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan', 'minkowski']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=Value_cv,scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    # Meilleur paramètre k
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")
    best_params = grid_search.best_params_
    # Meilleur modèle
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")

    # Entraînement du modèle k-NN avec le meilleur k
    knn = grid_search.best_estimator_
    knn.fit(X_train, Y_train)

    # Prédictions
    Y_pred = knn.predict(X_test)

    # Précision sur l'ensemble d'entraînement
    y_train_pred = knn.predict(X_train)
    train_accuracy = accuracy_score(Y_train, y_train_pred)

    # Évaluation des performances
    valid_labels = encoder.transform(encoder.classes_)  # Classes valides dans l'ordre des encodages
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, labels=valid_labels, target_names=encoder.classes_)

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print("\nClassification Report:\n")
    print(report)
    print("\n")
  
    # Matrix de confusion
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig_confusion = plt.gcf()
    plt.show()

    # Courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(knn, X_pca, Y_resampled, cv=Value_cv, scoring='accuracy')
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
    pred_probs = knn.predict_proba(X_test)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(valid_classes):
        sns.kdeplot(pred_probs[:, i], label=f'Class {class_name}')
    plt.title('Probability Distributions by Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    fig_Predict = plt.gcf()
    plt.show()

    if Save_Model:
        Smote = True
        # Définir les graphiques à sauvegarder
        images=[]    
        images.append(("Confusion Matrix", fig_confusion))
        images.append(("Learning Curve", fig_CurveLearning))
        images.append(("Probability Distributions by Class", fig_Predict))  

        save_knn_results(Smote,PCA_Number,best_params,knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)
