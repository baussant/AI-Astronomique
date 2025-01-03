# Evetools.py

import pandas as pd

class TriData:
    def __init__(self, file_path):
        """
        Initialise l'objet avec le chemin du fichier CSV et crée les quatre DataFrames en fonction des conditions.
        
        :param file_path: Chemin vers le fichier CSV.
        """
        self.file_path = file_path
        
        try:
            # Tentative de lecture du fichier CSV
            self.data = pd.read_csv(file_path)
            self.data = self.data.drop(columns=['TypeCore'], errors='ignore')
            # self.data['TypeCoreName'] = self.data['TypeCoreName'].astype(str)
            self.data = convert_columns_to_numeric(self.data)
            
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{file_path}' est introuvable.")
            self.data = None
        except pd.errors.EmptyDataError:
            print(f"Erreur : Le fichier '{file_path}' est vide.")
            self.data = None
        except pd.errors.ParserError:
            print(f"Erreur : Le fichier '{file_path}' contient des erreurs et ne peut pas être lu.")
            self.data = None
        except Exception as e:
            print(f"Erreur inconnue lors de l'ouverture du fichier '{file_path}': {e}")
            self.data = None

        # Initialisation des DataFrames si les données ont bien été chargées
        if self.data is not None:
      
            self.df_moon, self.df_planet, self.df_sun, self.df_asteroid = self.split_dataframes()
            # Nettoyage des colonnes vides ou nulles
            self.df_moon = self.clean_dataframe(self.df_moon)
            self.df_planet = self.clean_dataframe(self.df_planet)
            self.df_sun = self.clean_dataframe(self.df_sun)
            self.df_asteroid = self.clean_dataframe(self.df_asteroid)
        else:
            # Si les données sont None, les DataFrames seront également None
            self.df_moon = self.df_planet = self.df_sun = self.df_asteroid = None

    def split_dataframes(self):
        """
        Divise les données en quatre DataFrames selon les conditions spécifiées pour TypeCoreName.
        
        :return: Quatre DataFrames (df_moon, df_planet, df_sun, df_asteroid)
        """
        # Condition 1: 'TypeCoreName' est exactement 'Moon'

        df_moon = self.data[self.data['TypeCoreName'] == 'Moon']
        df_moon = df_moon.drop(columns=['TypeCoreName'], errors='ignore')
        # Condition 2: 'TypeCoreName' contient 'Planet' (insensible à la casse)
        df_planet = self.data[self.data['TypeCoreName'].str.contains('Planet', case=False, na=False)]
        # df_planet = df_planet.drop(columns=['TypeCoreName'], errors='ignore')
        # Condition 3: 'TypeCoreName' contient 'Sun' (insensible à la casse)
        df_sun = self.data[self.data['TypeCoreName'].str.contains('Sun', case=False, na=False)]
        df_sun = df_sun.drop(columns=['TypeCoreName'], errors='ignore')
        # Condition 4: 'TypeCoreName' contient 'Asteroid' (insensible à la casse)
        df_asteroid = self.data[self.data['TypeCoreName'].str.contains('Asteroid', case=False, na=False)]
        df_asteroid = df_asteroid.drop(columns=['TypeCoreName'], errors='ignore')
        return df_moon, df_planet, df_sun, df_asteroid

    def clean_dataframe(self, df):
        """
        Supprime les colonnes où toutes les valeurs sont nulles ou égales à zéro.
        
        :param df: Le DataFrame à nettoyer.
        :return: Le DataFrame nettoyé.
        """
        if df is not None:
            # Supprime les colonnes où toutes les valeurs sont nulles
            df = df.dropna(axis=1, how='all')

            # Supprime les colonnes où toutes les valeurs sont égales à zéro
            df = df.loc[:, (df != 0).any(axis=0)]
            
        
        return df

    def get_dataframes(self):
        """
        Retourne les quatre DataFrames si les données ont été chargées correctement.
        
        :return: Quatre DataFrames (df_moon, df_planet, df_sun, df_asteroid) ou None si le fichier n'a pas pu être chargé.
        """
        if self.data is None:
            print("Les DataFrames ne peuvent pas être renvoyés car le fichier CSV n'a pas été chargé correctement.")
            return None, None, None, None
        return self.df_moon, self.df_planet, self.df_sun, self.df_asteroid

def convert_columns_to_numeric(df, exclude_columns=None):
    """
    Convertit les colonnes contenant des valeurs numériques en type float, 
    en excluant certaines colonnes spécifiées.
    
    :param df: DataFrame pandas.
    :param exclude_columns: Liste des colonnes à exclure de la conversion.
    :return: DataFrame avec les colonnes converties.
    """
    if exclude_columns is None:
        exclude_columns = ['TypeCoreName','itemName']
    
    for col in df.columns:
        # Vérifier si la colonne est de type 'object' et non exclue
        if col not in exclude_columns and df[col].dtype == 'object':
            try:
                # Convertir les valeurs numériques en float, remplacer les erreurs par NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"✅ Colonne '{col}' convertie en numérique.")
            except Exception as e:
                print(f"❌ Erreur pour la colonne '{col}': {e}")
    return df