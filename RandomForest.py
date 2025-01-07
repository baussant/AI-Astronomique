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
from Graphique import Affichage_accuracy, Affichage_confusion_matrix, Affichage_roc_curve,Affichage_courbe_apprentissage,Affichage_proba2
from EcritureCSV import create_csv
from ModelHistory import save_rf_results,save_rf_results2
from ToolsAI import Calcul_XX_YY, split_data, apply_smote, apply_pca,sampling

def RandomForest2(sampling_strategy,sampling_number,smote_status,data,  grid_data, cv_data, X_Chara, Y_Target,Echantillon_min,Save_Model):

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
    # X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test=split_data(XX, YY)

    # Application de SMOTE pour équilibrer les classes
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
    fig_proba = Affichage_proba2(y_test_proba,valid_classes)
    

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
        Solver_Use='N/A'
        variance=0
        #save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,model, scaler, pca, PCA_State, grid_data , report,report_train, images,cv_data, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF")
        save_rf_results2(sampling_strategy,sampling_number,Smote,Best_parameter,model, grid_data , report,report_train, images,cv_data, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy, output_folder="RF")