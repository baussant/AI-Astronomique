import re # For regular expressions
import string
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import psutil
from psutil import process_iter

def process_and_plot_graphs2(Graph_Moons_Dict, Label_Curve_Dict, degree):
    """
    Crée les polynômes, calcule les déviations standard et génère les graphiques superposés.
    
    Parameters:
    - Graph_Moons_Dict (dict): Dictionnaire contenant les DataFrames filtrés.
    - Label_Curve_Dict (dict): Dictionnaire contenant les labels associés aux courbes.
    - degree (int): Degré du polynôme pour l'ajustement.

    Returns:
    - plot: Graphique combiné avec hvplot.
    """
    combined_plot = None
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for idx, (key, df) in enumerate(Graph_Moons_Dict.items(), start=1):
        # Sélectionner les colonnes 'radius' et 'density'
        df_filtered = df[['radius', 'density']]
 
        label = Label_Curve_Dict[str(idx)]
       
        
        # Ajustement du polynôme
        coeffs = np.polyfit(df_filtered['radius'], df_filtered['density'], degree)
        x_poly = np.linspace(df_filtered['radius'].min(), df_filtered['radius'].max(), 500)
        y_poly = np.polyval(coeffs, x_poly)
        
        # Calcul des résidus et de la déviation standard
        y_pred = np.polyval(coeffs, df_filtered['radius'])
        residuals = df_filtered['density'] - y_pred
        std_dev = np.std(residuals)
        
        # Calcul des bornes
        y_upper = y_poly + std_dev
        y_lower = y_poly - std_dev
        
        # Attribuer une couleur unique
        color = colors[(idx - 1) % len(colors)] 

        # Création des graphiques
        scatter = df_filtered.hvplot.scatter(x='radius', y='density', color='black', size=10, alpha=0.5, label=label)
        poly_curve = pd.DataFrame({'radius': x_poly, 'density': y_poly}).hvplot.line(
            x='radius', y='density', color=color, line_width=2, label=f"Polyno : {label} Std dév. : {std_dev}"
        )
        
        # Combiner les graphiques
        current_plot = scatter * poly_curve
        if combined_plot is None:
            combined_plot = current_plot
        else:
            combined_plot *= current_plot

        
        # Afficher la déviation standard
        print(f"Standard déviation {label} : {std_dev}\n")
    
    return combined_plot



def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def print_evaluate(true, predicted,label="-"):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(true, predicted)
    print(f"---------{label}----------")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Square: {r2}")

def prepare_data_graph(data,data_filter,labelx,labely):
 
    def_data = data[data['TypeCoreName'].str.contains(data_filter, case=False, na=False)]
    def_data = def_data.drop(columns=['TypeCoreName'])
    return def_data[[labelx,labely]]

