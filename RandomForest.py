import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import seaborn as sns
import xgboost as xgb
import GPUtil
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve,auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV,learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder,label_binarize
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from utils import  print_evaluate
from Graphique import Affichage_accuracy, Affichage_confusion_matrix, Affichage_roc_curve,Affichage_courbe_apprentissage,Affichage_proba
from EcritureCSV import create_csv
from ModelHistory import save_rf_results

def check_gpu_availability():
    print("Vérification des GPU disponibles...")
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("Aucun GPU détecté. Vérifiez vos pilotes et matériel.")
    else:
        for gpu in gpus:
            print(f"Carte GPU détectée : {gpu.name}, Utilisation : {gpu.load*100:.1f}%, Mémoire : {gpu.memoryFree}MB libre / {gpu.memoryTotal}MB total")

# Fonction pour vérifier l'utilisation du GPU avec XGBoost

def check_xgboost_gpu():
    try:
        print("Vérification de la configuration GPU pour XGBoost...")
        model = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        params = model.get_params()
        if params.get("tree_method") == "gpu_hist":
            print("XGBoost est configuré pour utiliser le GPU.")
        else:
            print("XGBoost n'est pas configuré pour le GPU. Vérifiez vos paramètres.")
    except Exception as e:
        print("Erreur lors de la vérification GPU pour XGBoost :", e)

# Mise à jour du mécanisme de sélection GPU pour XGBoost
def get_xgb_device():
    try:
        print("Détection du périphérique pour XGBoost...")
        gpus = GPUtil.getGPUs()
        if gpus:
            print("GPU détecté. Configuration pour XGBoost.")
            return "gpu_hist"
        else:
            print("GPU non disponible, basculement sur le CPU.")
            return "auto"
    except Exception as e:
        print(f"Erreur lors de la vérification GPU : {e}. Basculement sur le CPU.")
        return "auto"       

def RF_PCA2(PCA_State,data, variance, grid_data, cv_data, X_Chara, Y_Target, XGB_Use, Solver_Use,Echantillon_min,Save_Model, classification=True):

    # Filtrer les classes avec suffisamment de données
    class_counts = data['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    data = data[data['TypeCoreName'].isin(valid_classes)]

    # Calcul des données d'entrée et de sortie
    XX, YY, encoder = Calcul_XX_YY(data, Y_Target, X_Chara)
    
    print(f"------- Entête Y prise en compte ---------------------------")
    for colonne in XX.columns:
        print(f"{colonne}", end=" | ")
    print("\n------------------------------------------------------------")

    # Division des données
    X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Validation et ajustement dynamique de components_data
    max_components = X_train_scaled.shape[1]
    if not isinstance(variance, int) or variance <= 0 or variance > max_components:
        # Ajustement basé sur components_data de la variance expliquée
        pca_temp = PCA(svd_solver=Solver_Use, random_state=42)
        pca_temp.fit(X_train_scaled)
        explained_variance_ratio_cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        components_data_new = np.argmax(explained_variance_ratio_cumsum >= variance) + 1
        print(f"Nombre de composantes ajusté automatiquement à : {components_data_new} pour couvrir {variance}% de la variance expliquée.")

    # Application de la PCA
    try:
        pca = PCA(n_components=components_data_new, svd_solver=Solver_Use, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"Nombre de composantes PCA retenues : {components_data_new}")
    except ValueError as e:
        raise ValueError(f"Erreur avec PCA : {e}")

    # Affichage de la variance expliquée cumulée
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
    plt.axhline(y=variance, color='r', linestyle='--', label=f"{variance}% de variance expliquée")
    plt.xlabel('Composantes PCA')
    plt.ylabel('Variance expliquée cumulée')
    plt.title('Variance expliquée cumulée par les composantes PCA')
    plt.legend()
    plt.show()

    # GPU fallback mechanism
    xgb_tree_method = get_xgb_device()

    # Choix du modèle (Régression ou Classification)
    if classification:
        if not XGB_Use:
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            model = XGBClassifier(tree_method=xgb_tree_method, 
                              predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                              random_state=42)
        scoring_metric = 'accuracy'
    else:
        if not XGB_Use:
            model = RandomForestRegressor(random_state=42)
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            model = XGBRegressor(tree_method=xgb_tree_method, 
                             predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                             random_state=42)
        scoring_metric = 'neg_mean_squared_error'

    # Optimisation des hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_data,
        cv=cv_data,
        scoring=scoring_metric,
        n_jobs=-1
    )
    grid_search.fit(X_train_pca, Y_train)

    # Meilleur modèle et ses hyperparamètres
    best_model = grid_search.best_estimator_
    Best_parameter = grid_search.best_params_
    print("Meilleurs hyperparamètres :", Best_parameter)

    # Importance des caractéristiques si disponible
    if hasattr(best_model, 'feature_importances_'):
        pca_feature_names = [f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
        feature_importances = best_model.feature_importances_
        plt.barh(pca_feature_names, feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Composantes PCA')
        plt.title('Importance des Composantes PCA (Forêt Aléatoire)')
        plt.show()
    else:
        print("Le modèle sélectionné ne fournit pas d'importances des caractéristiques.")

    # Prédictions sur les ensembles d'entraînement et de test
    Y_train_pred = best_model.predict(X_train_pca)
    Y_test_pred = best_model.predict(X_test_pca)

    # Redéfinition de Y_test en tableau 1D
    Y_test = np.array(Y_test).ravel()


    # Affichage des résultats
    report,report_train,images,accuracy,train_accuracy = Affichage(Y_test, Y_test_pred, classification, encoder, Y_train, Y_train_pred,best_model,X_train_pca,X_test,cv_data,valid_classes)  # Retour Report, image,accuracy,train_accuracy

    if Save_Model:
        PCA_State= True
        Smote= False
        save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,model, scaler, pca, PCA_State, grid_data , report,report_train, images,cv_data, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF")

def RF_PCA_SMOTE(PCA_State,data, variance, grid_data, cv_data, X_Chara, Y_Target, XGB_Use, Solver_Use,Echantillon_min,Save_Model, classification=True):

    # Filtrer les classes avec suffisamment de données
    class_counts = data['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    data = data[data['TypeCoreName'].isin(valid_classes)]

    # Calcul des données d'entrée et de sortie
    XX, YY, encoder = Calcul_XX_YY(data, Y_Target, X_Chara)
    
    print(f"------- Entête Y prise en compte ---------------------------")
    for colonne in XX.columns:
        print(f"{colonne}", end=" | ")
    print("\n------------------------------------------------------------")

    # Division des données
    X_train, X_test, Y_train2, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    # Standardisation des données
    scaler = StandardScaler()
    X_train_scaled2 = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Application de SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_train_scaled, Y_train = smote.fit_resample(X_train_scaled2, Y_train2)
 
    # Validation et ajustement dynamique de components_data
    max_components = X_train_scaled.shape[1]
    if not isinstance(variance, int) or variance <= 0 or variance > max_components:
        # Ajustement basé sur components_data de la variance expliquée
        pca_temp = PCA(svd_solver=Solver_Use, random_state=42)
        pca_temp.fit(X_train_scaled)
        explained_variance_ratio_cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        components_data_new = np.argmax(explained_variance_ratio_cumsum >= variance) + 1
        print(f"Nombre de composantes ajusté automatiquement à : {components_data_new} pour couvrir {variance}% de la variance expliquée.")

    # Application de la PCA
    try:
        pca = PCA(n_components=components_data_new, svd_solver=Solver_Use, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"Nombre de composantes PCA retenues : {components_data_new}")
    except ValueError as e:
        raise ValueError(f"Erreur avec PCA : {e}")

    # Affichage de la variance expliquée cumulée
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
    plt.axhline(y=variance, color='r', linestyle='--', label=f"{variance}% de variance expliquée")
    plt.xlabel('Composantes PCA')
    plt.ylabel('Variance expliquée cumulée')
    plt.title('Variance expliquée cumulée par les composantes PCA')
    plt.legend()
    plt.show()



    # Choix du modèle (Régression ou Classification)
    if classification:
        if not XGB_Use:
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            # GPU fallback mechanism
            xgb_tree_method = get_xgb_device()
            model = XGBClassifier(tree_method=xgb_tree_method, 
                              predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                              random_state=42)
        scoring_metric = 'accuracy'
    else:
        if not XGB_Use:
            model = RandomForestRegressor(random_state=42)
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            # GPU fallback mechanism
            xgb_tree_method = get_xgb_device()
            model = XGBRegressor(tree_method=xgb_tree_method, 
                             predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                             random_state=42)
        scoring_metric = 'neg_mean_squared_error'

    # Optimisation des hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_data,
        cv=cv_data,
        scoring=scoring_metric,
        n_jobs=-1
    )
    grid_search.fit(X_train_pca, Y_train)

    # Meilleur modèle et ses hyperparamètres
    best_model = grid_search.best_estimator_
    Best_parameter = grid_search.best_params_
    print("Meilleurs hyperparamètres :", Best_parameter)

    # Importance des caractéristiques si disponible
    if hasattr(best_model, 'feature_importances_'):
        pca_feature_names = [f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]
        feature_importances = best_model.feature_importances_
        plt.barh(pca_feature_names, feature_importances)
        plt.xlabel('Importance')
        plt.ylabel('Composantes PCA')
        plt.title('Importance des Composantes PCA (Forêt Aléatoire)')
        plt.show()
    else:
        print("Le modèle sélectionné ne fournit pas d'importances des caractéristiques.")

    # Prédictions sur les ensembles d'entraînement et de test
    Y_train_pred = best_model.predict(X_train_pca)
    Y_test_pred = best_model.predict(X_test_pca)

    # Redéfinition de Y_test en tableau 1D
    Y_test = np.array(Y_test).ravel()


    # Affichage des résultats
    report,report_train,images,accuracy,train_accuracy = Affichage(Y_test, Y_test_pred, classification, encoder, Y_train, Y_train_pred,best_model,X_train_pca,X_test,cv_data,valid_classes)  # Retour Report, image,accuracy,train_accuracy

    if Save_Model:
        PCA_State= True
        Smote= True
    
        save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,model, scaler, pca, PCA_State, grid_data , report,report_train, images,cv_data, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF")

def RF_SMOTE(smote_status,data, variance, grid_data, cv_data, X_Chara, Y_Target, XGB_Use, Solver_Use,Echantillon_min,Save_Model, classification=True):

    # Filtrer les classes avec suffisamment de données
    class_counts = data['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    data = data[data['TypeCoreName'].isin(valid_classes)]

    # Calcul des données d'entrée et de sortie
    XX, YY, encoder = Calcul_XX_YY(data, Y_Target, X_Chara)
    
    print(f"------- Entête Y prise en compte ---------------------------")
    for colonne in XX.columns:
        print(f"{colonne}", end=" | ")
    print("\n------------------------------------------------------------")

    # Division des données
    X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    # Application de SMOTE pour équilibrer les classes
    if smote_status:
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    else:
        X_train_resampled = X_train 
        Y_train_resampled = Y_train   
 

    # Choix du modèle (Régression ou Classification)
    if classification:
        if not XGB_Use:
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            # GPU fallback mechanism
            xgb_tree_method = get_xgb_device()
            model = XGBClassifier(tree_method=xgb_tree_method, 
                              predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                              random_state=42)
        scoring_metric = 'accuracy'
    else:
        if not XGB_Use:
            model = RandomForestRegressor(random_state=42)
        else:
            # Suppression de check_gpu_usage() en raison d'un problème avec le module XGBoost
            # GPU fallback mechanism
            xgb_tree_method = get_xgb_device()
            model = XGBRegressor(tree_method=xgb_tree_method, 
                             predictor='gpu_predictor' if xgb_tree_method == 'gpu_hist' else 'cpu_predictor', 
                             random_state=42)
        scoring_metric = 'neg_mean_squared_error'

    # Optimisation des hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_data,
        cv=cv_data,
        scoring=scoring_metric,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, Y_train_resampled)

    # Meilleur modèle et ses hyperparamètres
    best_model = grid_search.best_estimator_
    Best_parameter = grid_search.best_params_
    print("Meilleurs hyperparamètres :", Best_parameter)

    # Prédictions sur les ensembles d'entraînement et de test
    Y_train_pred = best_model.predict(X_train_resampled)
    Y_test_pred = best_model.predict(X_test)

    # Redéfinition de Y_test en tableau 1D
    Y_test = np.array(Y_test).ravel()


    # Affichage des résultats
    report,report_train,images,accuracy,train_accuracy = Affichage(Y_test, Y_test_pred, classification, encoder, Y_train_resempled, Y_train_pred,best_model,X_train_resempled,X_test,cv_data,valid_classes)  # Retour Report, image,accuracy,train_accuracy

    if Save_Model:
        PCA_State= False
        Smote= smote_status
        components_data_new=0
        scaler=0
        pca=0
        save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,model, scaler, pca, PCA_State, grid_data , report,report_train, images,cv_data, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF")

def RF_SMOTE2(smote_status,data,  grid_data, cv_data, X_Chara, Y_Target,Echantillon_min,Save_Model):


    # Filtrer les classes avec suffisamment de données
    class_counts = data['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    data = data[data['TypeCoreName'].isin(valid_classes)]

    # Calcul des données d'entrée et de sortie
    XX, YY, encoder = Calcul_XX_YY(data, Y_Target, X_Chara)
    
    print(f"------- Entête Y prise en compte ---------------------------")
    for colonne in XX.columns:
        print(f"{colonne}", end=" | ")
    print("\n------------------------------------------------------------")

    # Division des données
    X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    # Application de SMOTE pour équilibrer les classes
    if smote_status:
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
        # Décodage pour affichage
        Y_resampled_decoded = pd.DataFrame({'TypeCoreName': encoder.inverse_transform(Y_train_resampled)})
        print("----------------  Répartition Données après SMOTE --------------\n")
        typecore_count = Y_resampled_decoded['TypeCoreName'].value_counts()
        typecore_table = typecore_count.to_frame().T
        print(typecore_table)
    else:
        X_train_resampled = X_train 
        Y_train_resampled = Y_train   
 
    # Choix du modèle (Classification)
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    scoring_metric = 'accuracy'

    # Optimisation des hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid_data,
        cv=cv_data,
        scoring=scoring_metric,
        n_jobs=-1
    )
    grid_search.fit(X_train_resampled, Y_train_resampled)

    # Meilleur modèle et ses hyperparamètres
    best_model = grid_search.best_estimator_
    Best_parameter = grid_search.best_params_
    print("Meilleurs hyperparamètres :", Best_parameter)

    # Prédictions sur les ensembles d'entraînement et de test
    Y_train_pred = best_model.predict(X_train_resampled)
    Y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    # Redéfinition de Y_test en tableau 1D
    Y_test = np.array(Y_test).ravel()

    report,report_train,accuracy,train_accuracy =  Affichage_accuracy(Y_test, Y_test_pred,Y_train_resampled, Y_train_pred,encoder)
    fig_confusion = Affichage_confusion_matrix(Y_test, Y_test_pred, encoder)
    fig_learning_curve = Affichage_courbe_apprentissage(best_model, X_train_resampled, Y_train_resampled, cv_data)
    fig_roc = Affichage_roc_curve(Y_test, Y_test_pred, encoder, best_model, X_test)
    fig_proba = Affichage_proba(y_test_proba)
    

    # Affichage des résultats
    #report,report_train,images,accuracy,train_accuracy = Affichage(Y_test, Y_test_pred, classification, encoder, Y_train_resempled, Y_train_pred,best_model,X_train_resempled,X_test,cv_data,valid_classes)  # Retour Report, image,accuracy,train_accuracy
    images=[]    
    images.append(("Confusion Matrix", fig_confusion))
    images.append(("ROC Curve",  fig_roc))
    images.append(("Learning Curve", fig_learning_curve))
    images.append(("Probability", fig_proba))

    if Save_Model:
        PCA_State= False
        Smote= smote_status
        components_data_new=0
        scaler=0
        pca=0
        XGB_Use=False
        classification=True
        Solver_Use='N/A'
        variance=0
        save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,model, scaler, pca, PCA_State, grid_data , report,report_train, images,cv_data, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF")


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

def Affichage(Y_test,Y_test_pred,classification,encoder,Y_train,Y_train_pred,best_model,X_train_pca,X_test,cv_data,valid_classes):

    # Évaluation des performances
    valid_labels = encoder.transform(encoder.classes_)  # Classes valides dans l'ordre des encodages
    accuracy = accuracy_score(Y_test, Y_test_pred)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)

    # Affichage des résultats
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print("\nClassification Report:\n")
    
    # Graphique : Valeurs réelles vs prédictions
    plt.figure()
    plt.scatter(Y_test, Y_test_pred, alpha=0.5, label='Prédictions')
    #plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Référence (y=x)')
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label=valid_labels, target_names=encoder.classes_)

    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title('Valeurs Réelles vs Prédites (Forêt Aléatoire avec PCA)')
    plt.legend()
    fig_scatter = plt.gcf()
    plt.show()
    
    # Distribution des erreurs résiduelles
    # residuals = Y_test.values.ravel() - Y_test_pred
    residuals = Y_test - Y_test_pred

    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Erreur résiduelle')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs résiduelles')
    fig_histogram = plt.gcf() 
    plt.show()

    # Évaluation du modèle
    
    if classification:
        # print(classification_report(Y_test, Y_test_pred,labels=valid_labels, target_names=encoder.classes_))
        Report_test = classification_report(Y_test, Y_test_pred, target_names=encoder.classes_)
        print(f"\n----------Rapport de test-----\n")
        print(Report_test)
        print(f"\n----------Rapport de train-----\n")
        Report_train = classification_report(Y_train, Y_train_pred,labels=valid_labels, target_names=encoder.classes_)
        print(Report_train)
        cm = confusion_matrix(Y_test, Y_test_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Prédictions')
        plt.ylabel('Véritables')
        plt.title('Matrice de confusion')
        fig_confusion = plt.gcf()
        plt.show()
    else:
        Report_test = None
        Report_train = None
        fig_confusion = None
        print_evaluate(Y_test, Y_test_pred, 'Test set evaluation')
        print(f"\n")
        print("R² sur le jeu d'entraînement : ", best_model.score(X_train_pca, Y_train))

    # Courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_pca, Y_train, cv=cv_data, scoring='accuracy')
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
    pred_probs = best_model.predict_proba(X_test)
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
    y_test_binarized = label_binarize(Y_test, classes=range(len(encoder.classes_)))
    n_classes = y_test_binarized.shape[1]

    # Courbe ROC
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonale
    # Calcul des courbes ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Classe {encoder.classes_[i]} (AUC = {roc_auc:.2f})")
    plt.title("Courbes ROC Multiclasses - Random Forest")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    fig_roc = plt.gcf()
    plt.show()
    
         
    images=[]    
    if classification:
        images.append(("Confusion Matrix", fig_confusion))
    images.append(("Valeurs Réelles vs Prédites (Forêt Aléatoire avec PCA)", fig_scatter))
    images.append(("Distribution des erreurs résiduelles", fig_histogram))   
    images.append(("Courbe d'apprentissage", fig_CurveLearning))
    images.append(("Distribution des probabilités de Prédiction", fig_Predict))
    images.append(("Courbe ROC", fig_roc)) 

    # Retour Report, image,accuracy,train_accuracy
    return Report_test,Report_train, images, accuracy,train_accuracy
        