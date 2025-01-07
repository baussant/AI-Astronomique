import sqlite3
import os
from typing import Any

def create_or_open_database(db_name: str) -> sqlite3.Connection:
    """
    Crée une base de données SQLite ou l'ouvre si elle existe déjà.

    :param db_name: Nom de la base de données.
    :return: Connexion à la base de données.
    """
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))  # Chemin du répertoire racine du projet
        db_path = os.path.join(project_root, db_name)
        connection = sqlite3.connect(db_path)
        print(f"Base de données '{db_name}' ouverte avec succès dans '{db_path}'.")
        return connection
    except sqlite3.Error as e:
        print(f"Erreur lors de l'ouverture de la base de données : {e}")
        raise

def create_rf_data_table(connection: sqlite3.Connection):
    """
    Crée la table RF_data avec les colonnes spécifiées si elle n'existe pas déjà.

    :param connection: Connexion à la base de données SQLite.
    """
    create_table_query ="""CREATE TABLE IF NOT EXISTS RF_data (
        Smote BOOLEAN,
        CV INTEGER,
        Sampling_Strategy BOOLEAN,
        Sampling_Number TEXT,
        Max_depth INTEGER,
        Max_features TEXT,
        Min_sample_leaf REAL,
        Min_sample_split REAL,
        N_estimator INTEGER,
        Accuracy REAL,
        Accuracy_train REAL,
        PARAMS TEXT,
        FileName TEXT
        
    );"""
   # PARAMS TEXT
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'RF_data' créée ou déjà existante.")
    except sqlite3.Error as e:
        print(f"Erreur lors de la création de la table : {e}")
        raise

def create_knn_data_table(connection: sqlite3.Connection):
    """
    Crée la table RF_data avec les colonnes spécifiées si elle n'existe pas déjà.

    :param connection: Connexion à la base de données SQLite.
    (knn,scaler,pca if PCA_State else None,PCA_State,param_grid,report,images,Value_cv,variance,Y_Target,X_Chara,neighbor_max,Echantillon_min,accuracy,train_accuracy)
    """
    create_table_query ="""CREATE TABLE IF NOT EXISTS KNN_data (
        Smote BOOLEAN,
        Sampling_Strategy TEXT,
        Sampling_Number TEXT,
        CV INTEGER,
        Variance REAL,
        Neighbor_Max INTEGER,
        Neighbor INTEGER,
        Weights TEXT,
        Metric TEXT,
        Accuracy REAL,
        Accuracy_train REAL,
        PCA BOOLEAN,
        PCA_Number INTEGER,
        PARAMS TEXT,
        FileName TEXT
        
    );"""
   # PARAMS TEXT
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'knn_data' créée ou déjà existante.")
    except sqlite3.Error as e:
        print(f"Erreur lors de la création de la table : {e}")
        raise

def create_dp_data_table(connection: sqlite3.Connection):
    """
    Crée la table DP_data avec les colonnes spécifiées si elle n'existe pas déjà.

    :param connection: Connexion à la base de données SQLite.
    """
    create_table_query ="""CREATE TABLE IF NOT EXISTS DP_data (
        Smote BOOLEAN,
        Accuracy REAL,
        Accuracy_train REAL,
        Echantillon_min INTEGER,
        Weight_Class BOOLEAN,
        Adjust_Factor REAL,
        Epoque INTEGER,
        Batch_size_nbr INTEGER,
        Learning_Rate REAL,
        FileName TEXT
        
    );"""
   # PARAMS TEXT
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()
        print("Table 'DP_data' créée ou déjà existante.")
    except sqlite3.Error as e:
        print(f"Erreur lors de la création de la table : {e}")
        raise    

def insert_rf_data(connection: sqlite3.Connection, data: dict[str, Any]):
    """
    Insère des données dans la table RF_data.

    :param connection: Connexion à la base de données SQLite.
    :param data: Dictionnaire contenant les données à insérer.
    """
    create_rf_data_table(connection) 

    insert_query = """
    INSERT INTO RF_data (Smote,CV, Sampling_Strategy,Sampling_number,Max_depth, Max_features, Min_sample_leaf, 
                         Min_sample_split, N_estimator, Accuracy, Accuracy_train, PARAMS,FileName)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?); """
    
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (
            data['Smote'],
            data['CV'],
            data['Sampling_Strategy'],
            data['Sampling_Number'],
            data['Max_depth'],
            data['Max_features'],
            data['Min_sample_leaf'],
            data['Min_sample_split'],
            data['N_estimator'],
            data['Accuracy'],
            data['Accuracy_train'],
            data['PARAMS'],
            data['FileName']
            
        ))
        connection.commit()
        print("Données insérées avec succès dans la table RF_data.")
    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion des données : {e}")
        raise

def insert_dp_data(connection: sqlite3.Connection, data: dict[str, Any]):
    """
    Insère des données dans la table dp_data.

    :param connection: Connexion à la base de données SQLite.
    :param data: Dictionnaire contenant les données à insérer.
    """
    create_dp_data_table(connection) 

    insert_query = """
    INSERT INTO DP_data (Smote,Accuracy, Accuracy_train,Echantillon_min,Weight_Class,Adjust_Factor,Epoque,Batch_size_nbr,Learning_Rate, FileName)
    VALUES (?, ?, ?,?,?,?,?,?,?,?); """
    
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (
            data['Smote'],
            data['Accuracy'],
            data['Accuracy_train'],
            data['Echantillon_min'],
            data['Weight_Class'],
            data['Adjust_Factor'],
            data['Epoque'],
            data['Batch_size_nbr'],
            data['Learning_Rate'],
            data['FileName']
            
        ))
        connection.commit()
        print("Données insérées avec succès dans la table DP_data.")
    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion des données : {e}")
        raise    

def insert_knn_data(connection: sqlite3.Connection, data: dict[str, Any]):
    """
    Insère des données dans la table knn_data.

    :param connection: Connexion à la base de données SQLite.
    :param data: Dictionnaire contenant les données à insérer.
            CV,Variance REAL,Neighbor_Max,Neighbor,Weights,Metric,Accuracy,Accuracy_train,PCA,PCA_Number,PARAMS,FileName 
    """
    create_knn_data_table(connection) 

    insert_query = """
    INSERT INTO KNN_data (Smote,Sampling_Strategy,Sampling_Number,CV,Variance,Neighbor_Max,Neighbor,Weights,Metric,
                        Accuracy,Accuracy_train,PCA,PCA_Number,PARAMS,FileName)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?,?); """
    
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (
            data['Smote'],
            data['Sampling_Strategy'],
            data['Sampling_Number'],
            data['CV'],
            data['Variance'],
            data['Neighbor_Max'],
            data['Neighbor'],
            data['Weights'],
            data['Metric'],
            data['Accuracy'],
            data['Accuracy_train'],
            data['PCA'],
            data['PCA_Number'],
            data['PARAMS'],
            data['FileName']
            
        ))
        connection.commit()
        print("Données insérées avec succès dans la table knn_data.")
    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion des données : {e}")
        raise    

if __name__ == "__main__":
    db_connection = create_or_open_database("data.db")
    create_rf_data_table(db_connection)
    create_knn_data_table(db_connection)

          
