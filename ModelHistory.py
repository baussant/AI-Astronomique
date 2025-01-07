import os
import zipfile
import joblib
import datetime
import matplotlib.pyplot as plt
from SqliteDB import insert_rf_data,create_or_open_database,insert_knn_data,insert_dp_data

def save_knn_results(sampling_number,sampling_strategy,Smote,PCA_Number,best_params,knn_model, scaler, pca, PCA_State, params, report, images,Value_cv, variance, Y_Target, X_Chara, neighbor_max, Echantillon_min,accuracy,train_accuracy, output_folder="KNN"):
    """
    Enregistre les modèles, graphiques, et rapports dans un fichier ZIP.
    
    Parameters:
    - knn_model : Le modèle KNN entraîné.
    - scaler : L'objet StandardScaler utilisé pour la normalisation.
    - pca : L'objet PCA utilisé si PCA_State est True.
    - PCA_State : Booléen, indique si PCA a été utilisé.
    - params : Dictionnaire des paramètres utilisés pour l'entraînement.
    - report : Le rapport de classification sous forme de chaîne.
    - images : Liste de tuples contenant les graphiques (nom_fichier, figure).
    - output_folder : Nom du dossier où enregistrer les fichiers.
    - Value_cv : Nombre de validations croisées.
    - variance : Variance retenue par PCA.
    - Y_Target : Cible d'entraînement.
    - X_Chara : Caractéristiques d'entraînement.
    - neighbor_max : Nombre maximal de voisins testé.
    - Echantillon_min : Nombre minimal d'échantillons pour inclure une classe.
    """
    # 1. Création du répertoire 'KNN' s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # 2. Définir le nom du fichier ZIP de manière dynamique
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pca_status = "With-PCA" if PCA_State else "Without-PCA"
    #zip_filename = os.path.join(output_folder, f"{pca_status}-{date_str}.zip")
    zip_filename = os.path.join(output_folder, f"{pca_status}-CV={Value_cv}-Variance={variance}-{date_str}.zip")

    # 3. Enregistrer les modèles dans des fichiers .pkl
    knn_filename = os.path.join(output_folder, "knn_model.pkl")
    scaler_filename = os.path.join(output_folder, "scaler.pkl")
    joblib.dump(knn_model, knn_filename)
    joblib.dump(scaler, scaler_filename)
    if PCA_State and pca is not None:
        pca_filename = os.path.join(output_folder, "pca_transformer.pkl")
        joblib.dump(pca, pca_filename)
    else:
        pca_filename = None

    # 4. Enregistrer le fichier texte avec les paramètres et le rapport
    params_filename = os.path.join(output_folder, "training_report.txt")
    with open(params_filename, "w") as f:
        f.write("===== PCA utilisé =====\n")
        f.write("PCA : {PCA_Number}\n")
        f.write("===== Paramètres d'Entraînement =====\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n===== Rapport de Classification =====\n")
        f.write(f"Accuracy : {accuracy}\n")
        f.write(f"Train Accuracy : {train_accuracy}\n\n")
        f.write(report)
    
    # 5. Enregistrer un fichier texte 'parameter' avec des détails supplémentaires
    parameter_filename = os.path.join(output_folder, "parameters.txt")
    with open(parameter_filename, "w") as f:
        f.write("===== Paramètres Complémentaires =====\n")
        f.write(f"Value_cv: {Value_cv}\n")
        f.write(f"Variance (PCA): {variance}\n")
        f.write(f"Y_Target: {Y_Target}\n")
        f.write(f"X_Chara: {X_Chara}\n")
        f.write(f"Neighbor_max: {neighbor_max}\n")
        f.write(f"PCA_State: {PCA_State}\n")
        f.write(f"Echantillon_min: {Echantillon_min}\n")
        f.write(f"Params: {params}\n")
        f.write(f"Smote: {Smote}\n")
        f.write(f"Sampling_strategy: {sampling_strategy}\n")
        f.write(f"Sampling_number: {sampling_number}\n")

    # 6. Enregistrer les graphiques dans des fichiers images
    image_files = []
    for name, fig in images:
        image_path = os.path.join(output_folder, f"{name}.png")
        fig.savefig(image_path)
        image_files.append(image_path)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

    # 7. Créer un fichier ZIP contenant tous les fichiers
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(knn_filename, os.path.basename(knn_filename))
        zipf.write(scaler_filename, os.path.basename(scaler_filename))
        if pca_filename:
            zipf.write(pca_filename, os.path.basename(pca_filename))
        zipf.write(params_filename, os.path.basename(params_filename))
        zipf.write(parameter_filename, os.path.basename(parameter_filename))
        for img_file in image_files:
            zipf.write(img_file, os.path.basename(img_file))

    print(f"Fichiers sauvegardés dans {zip_filename}")

    # 8. Nettoyage des fichiers intermédiaires (optionnel)
    os.remove(knn_filename)
    os.remove(scaler_filename)
    if pca_filename:
        os.remove(pca_filename)
    os.remove(params_filename)
    os.remove(parameter_filename)
    for img_file in image_files:
        os.remove(img_file)

    params_to_text = ""

    for key,value in params.items():
        params_to_text +=f"{key}:{value},"

    sample_data = {
        'Smote': Smote,
        'Sampling_Strategy': sampling_strategy,
        'Sampling_Number': sampling_number,
        'CV':Value_cv,
        'Variance':variance,
        'Neighbor_Max': neighbor_max,
        'Neighbor' : best_params['n_neighbors'],
        'Weights' : best_params['weights'],
        'Metric' : best_params['metric'],
        'Accuracy': accuracy,
        'Accuracy_train': train_accuracy,
        'PCA': PCA_State,
        'PCA_Number': PCA_Number,
        'PARAMS' : params_to_text,
        'FileName':zip_filename
       
    }    

    db_connection = create_or_open_database("data.db")  
    insert_knn_data(db_connection,sample_data)
    db_connection.close()   


def save_rf_results(Smote,components_data_new,XGB_Use,Best_parameter,RF_model, scaler, pca, PCA_State, params, report_test,report_train, images,Value_cv, variance, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy,Solver_Use, output_folder="RF"):


    # 1. Création du répertoire 'KNN' s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # 2. Définir le nom du fichier ZIP de manière dynamique
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pca_status = "With-PCA" if PCA_State else "Without-PCA"
    #zip_filename = os.path.join(output_folder, f"{pca_status}-{date_str}.zip")
    zip_filename = os.path.join(output_folder, f"{pca_status}-CV={Value_cv}-Variance={variance}-{date_str}.zip")

    # 3. Enregistrer les modèles dans des fichiers .pkl
    rf_filename = os.path.join(output_folder, "rf_model.pkl")
    scaler_filename = os.path.join(output_folder, "scaler.pkl")
    joblib.dump(RF_model, rf_filename)
    joblib.dump(scaler, scaler_filename)
    if PCA_State and pca is not None:
        pca_filename = os.path.join(output_folder, "pca_transformer.pkl")
        joblib.dump(pca, pca_filename)
    else:
        pca_filename = None

    # 4. Enregistrer le fichier texte avec les paramètres et le rapport
    params_filename = os.path.join(output_folder, "training_report.txt")
    with open(params_filename, "w") as f:
        f.write("===== Paramètres d'Entraînement =====\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("==== Meilleurs Paramètres ========\n")
        f.write(f"Best Parameters : {Best_parameter}\n")
        f.write("\n===== Rapport de Classification =====\n")
        f.write(f"Accuracy : {accuracy}\n")
        f.write(f"Train Accuracy : {train_accuracy}\n\n")
        f.write("\n===== Rapport set de test =====\n")
        f.write(report_test)
        f.write("\n===== Rapport set de train =====\n")
        f.write(report_train)
    
    # 5. Enregistrer un fichier texte 'parameter' avec des détails supplémentaires
    parameter_filename = os.path.join(output_folder, "parameters.txt")
    with open(parameter_filename, "w") as f:
        f.write("===== Paramètres Complémentaires =====\n")
        f.write(f"Model: {RF_model}\n")  
        f.write(f"Solver: {Solver_Use}\n")
        f.write(f"CV: {Value_cv}\n")
        f.write(f"Variance (PCA): {variance}\n")
        f.write(f"Y Target: {Y_Target}\n")
        f.write(f"X Samples: {X_Chara}\n")
        f.write(f"PCA_State: {PCA_State}\n")
        f.write(f"Echantillon_min: {Echantillon_min}\n")
        f.write(f"Params: {params}\n")
        f.write(f"Smote: {Smote}\n")

    # 6. Enregistrer les graphiques dans des fichiers images
    image_files = []
    for name, fig in images:
        image_path = os.path.join(output_folder, f"{name}.png")
        fig.savefig(image_path)
        image_files.append(image_path)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

    # 7. Créer un fichier ZIP contenant tous les fichiers
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(rf_filename, os.path.basename(rf_filename))
        zipf.write(scaler_filename, os.path.basename(scaler_filename))
        if pca_filename:
            zipf.write(pca_filename, os.path.basename(pca_filename))
        zipf.write(params_filename, os.path.basename(params_filename))
        zipf.write(parameter_filename, os.path.basename(parameter_filename))
        for img_file in image_files:
            zipf.write(img_file, os.path.basename(img_file))

    print(f"Fichiers sauvegardés dans {zip_filename}")

    # 8. Nettoyage des fichiers intermédiaires (optionnel)
    os.remove(rf_filename)
    os.remove(scaler_filename)
    if pca_filename:
        os.remove(pca_filename)
    os.remove(params_filename)
    os.remove(parameter_filename)
    for img_file in image_files:
        os.remove(img_file)        

    params_to_text = ""

    for key,value in params.items():
        params_to_text +=f"{key}:{value},"


    sample_data = {
        'Smote': Smote,
        'CV':Value_cv,
        'Variance':variance,
        'Solver':Solver_Use,
        'Max_depth': Best_parameter['max_depth'],
        'Max_features': Best_parameter['max_features'],
        'Min_sample_leaf': Best_parameter['min_samples_leaf'],
        'Min_sample_split': Best_parameter['min_samples_split'],
        'N_estimator': Best_parameter['n_estimators'],
        'Accuracy': accuracy,
        'Accuracy_train': train_accuracy,
        'PCA': PCA_State,
        'PCA_Number': int(components_data_new),
        'XGB_USE': XGB_Use,
        'PARAMS' : params_to_text,
        'FileName':zip_filename
       
    }

    db_connection = create_or_open_database("data.db")  
    insert_rf_data(db_connection,sample_data)
    db_connection.close()  

def save_dp_results(Smote_States,report,model,scaler, accuracy, accuracy_train, images,Echantillon_min,Weight_Class,Adjust_Factor,Epoque,batch_size_nbr,Learning_Rate, output_folder="DP"):

    # 1. Création du répertoire 'KNN' s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # 2. Définir le nom du fichier ZIP de manière dynamique
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #zip_filename = os.path.join(output_folder, f"{pca_status}-{date_str}.zip")
    zip_filename = os.path.join(output_folder, f"DP-Epoch={Epoque},Batch_size={batch_size_nbr}-{date_str}.zip")

    # 3. Enregistrer les modèles dans des fichiers .pkl
    dp_filename = os.path.join(output_folder, "dp_model.pkl")
    scaler_filename = os.path.join(output_folder, "scaler.pkl")
    joblib.dump(model, dp_filename)
    joblib.dump(scaler, scaler_filename)

    # 4. Enregistrer le fichier texte avec les paramètres et le rapport
    params_filename = os.path.join(output_folder, "training_report.txt")
    with open(params_filename, "w") as f:
        f.write("\n===== Rapport de Classification =====\n")
        f.write(f"Accuracy : {accuracy}\n")
        f.write(f"Train Accuracy : {accuracy_train}\n\n")
        f.write("\n===== Rapport set de test =====\n")
        f.write(report)
    
    # 5. Enregistrer un fichier texte 'parameter' avec des détails supplémentaires
    parameter_filename = os.path.join(output_folder, "parameters.txt")
    with open(parameter_filename, "w") as f:
        f.write("===== Paramètres Complémentaires =====\n")
        f.write(f"Model: {model}\n") 
        f.write(f"Echantillon_min: {Echantillon_min}\n")
        f.write("\n===== Paramètres d'Entraînement =====\n")
        f.write(f"Weight_Class : {Weight_Class}\n")
        f.write(f"Adjust_Factor : {Adjust_Factor}\n")
        f.write(f"Epoque : {Epoque}\n")
        f.write(f"Batch_size_nbr : {batch_size_nbr}\n")
        f.write(f"Learning_Rate : {Learning_Rate}\n")
        f.write(f"Smote_States : {Smote_States}\n") 

    # 6. Enregistrer les graphiques dans des fichiers images
    image_files = []
    for name, fig in images:
        image_path = os.path.join(output_folder, f"{name}.png")
        fig.savefig(image_path)
        image_files.append(image_path)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

    # 7. Créer un fichier ZIP contenant tous les fichiers
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(dp_filename, os.path.basename(dp_filename))
        zipf.write(scaler_filename, os.path.basename(scaler_filename))
        zipf.write(params_filename, os.path.basename(params_filename))
        zipf.write(parameter_filename, os.path.basename(parameter_filename))
        for img_file in image_files:
            zipf.write(img_file, os.path.basename(img_file))

    print(f"Fichiers sauvegardés dans {zip_filename}")

    # 8. Nettoyage des fichiers intermédiaires (optionnel)
    os.remove(dp_filename)
    os.remove(scaler_filename)
    os.remove(params_filename)
    os.remove(parameter_filename)
    for img_file in image_files:
        os.remove(img_file)        

    sample_data = {
        'Smote': Smote_States,
        'Accuracy': accuracy,
        'Accuracy_train': accuracy_train,
        'Echantillon_min': Echantillon_min,
        'Weight_Class': Weight_Class,
        'Adjust_Factor': Adjust_Factor,
        'Epoque': Epoque,
        'Batch_size_nbr': batch_size_nbr,
        'Learning_Rate': Learning_Rate,
        'FileName':zip_filename
       
    }

    db_connection = create_or_open_database("data.db")  
    insert_dp_data(db_connection,sample_data)
    db_connection.close()      

def save_rf_results2(sampling_strategy,sampling_number,Smote,Best_parameter,model, grid_data , report_test,report_train, images,cv_data, Y_Target, X_Chara, Echantillon_min,accuracy,train_accuracy, output_folder="RF"):


    # 1. Création du répertoire 'KNN' s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # 2. Définir le nom du fichier ZIP de manière dynamique
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pca_status = "With-Smote" if Smote else "Without-Smote"
    #zip_filename = os.path.join(output_folder, f"{pca_status}-{date_str}.zip")
    zip_filename = os.path.join(output_folder, f"{pca_status}-CV={cv_data}-{date_str}.zip")

    # 3. Enregistrer les modèles dans des fichiers .pkl
    rf_filename = os.path.join(output_folder, "rf_model.pkl")
    joblib.dump(model, rf_filename)
  
    # 4. Enregistrer le fichier texte avec les paramètres et le rapport
    params_filename = os.path.join(output_folder, "training_report.txt")
    with open(params_filename, "w") as f:
        f.write("===== Paramètres d'Entraînement =====\n")
        for key, value in grid_data.items():
            f.write(f"{key}: {value}\n")
        f.write("==== Meilleurs Paramètres ========\n")
        f.write(f"Best Parameters : {Best_parameter}\n")
        f.write("\n===== Rapport de Classification =====\n")
        f.write(f"Accuracy : {accuracy}\n")
        f.write(f"Train Accuracy : {train_accuracy}\n\n")
        f.write("\n===== Rapport set de test =====\n")
        f.write(report_test)
        f.write("\n===== Rapport set de train =====\n")
        f.write(report_train)
    
    # 5. Enregistrer un fichier texte 'parameter' avec des détails supplémentaires
    parameter_filename = os.path.join(output_folder, "parameters.txt")
    with open(parameter_filename, "w") as f:
        f.write("===== Paramètres Complémentaires =====\n")
        f.write(f"Model: {model}\n")  
        f.write(f"Y Target: {Y_Target}\n")
        f.write(f"X Samples: {X_Chara}\n")
        f.write(f"Echantillon_min: {Echantillon_min}\n")
        f.write(f"Params: {grid_data}\n")
        f.write(f"Smote: {Smote}\n")
        f.write(f"Sampling_strategy: {sampling_strategy}\n")
        f.write(f"Sampling_number: {sampling_number}\n")

    # 6. Enregistrer les graphiques dans des fichiers images
    image_files = []
    for name, fig in images:
        image_path = os.path.join(output_folder, f"{name}.png")
        fig.savefig(image_path)
        image_files.append(image_path)
        plt.close(fig)  # Fermer la figure pour libérer la mémoire

    # 7. Créer un fichier ZIP contenant tous les fichiers
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(rf_filename, os.path.basename(rf_filename))
        zipf.write(params_filename, os.path.basename(params_filename))
        zipf.write(parameter_filename, os.path.basename(parameter_filename))
        for img_file in image_files:
            zipf.write(img_file, os.path.basename(img_file))

    print(f"Fichiers sauvegardés dans {zip_filename}")

    # 8. Nettoyage des fichiers intermédiaires (optionnel)
    os.remove(rf_filename)
    os.remove(params_filename)
    os.remove(parameter_filename)
    for img_file in image_files:
        os.remove(img_file)        

    params_to_text = ""

    for key,value in grid_data.items():
        params_to_text +=f"{key}:{value},"


    sample_data = {
        'Smote': Smote,
        'CV':cv_data,
        'Sampling_Strategy': sampling_strategy,
        'Sampling_Number': sampling_number,
        'Max_depth': Best_parameter['max_depth'],
        'Max_features': Best_parameter['max_features'],
        'Min_sample_leaf': Best_parameter['min_samples_leaf'],
        'Min_sample_split': Best_parameter['min_samples_split'],
        'N_estimator': Best_parameter['n_estimators'],
        'Accuracy': accuracy,
        'Accuracy_train': train_accuracy,
        'PARAMS' : params_to_text,
        'FileName':zip_filename
       
    }

    db_connection = create_or_open_database("data.db")  
    insert_rf_data(db_connection,sample_data)
    db_connection.close()  