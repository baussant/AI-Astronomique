import re # For regular expressions
import string
import os
import numpy as np
import pandas as pd 

def Nbr_Moons2(data):
    # Extraire les colonnes System et Planet directement
    data['System'] = data['itemName'].str.extract(r'^(\w+)')  # Extraire le système
    # data['Planet'] = data['itemName'].str.extract(r'^\w+\s([\w\s]+)\s-\sMoon')  # Extraire la planète Shana V - Moon 1
    data['Planet'] = data['itemName'].str.extract(r'^\w+\s([\w\s]+)\s')
    # Vérifier que les colonnes sont correctement extraites
    if data['System'].isnull().all() or data['Planet'].isnull().all():
        raise ValueError("L'extraction des systèmes ou des planètes a échoué. Vérifiez les données et les expressions régulières.")

    # Créer un nouveau DataFrame avec les colonnes extraites
    extracted_data = data[['System', 'Planet']]

    # Compter le nombre de lunes par système et planète
    moon_counts = extracted_data.groupby(['System', 'Planet']).size().reset_index(name='MoonCount')
    return moon_counts

def get_unique_planet_types(df):
    """
    Renvoie la liste des types de planètes distincts en fonction de la colonne 'TypeCoreName'.
    
    Parameters:
    - df (DataFrame): Le DataFrame contenant les données des planètes, avec une colonne 'TypeCoreName'.
    
    Returns:
    - List: Une liste des types de noyau de planète distincts.
    """
    if 'TypeCoreName' in df.columns:
        unique_types = df['TypeCoreName'].unique()  # Récupérer les types distincts
        return list(unique_types)  # Retourner la liste des types
    else:
        raise ValueError("La colonne 'TypeCoreName' n'existe pas dans le DataFrame.")