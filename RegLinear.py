import pandas as pd
import time
import hvplot.pandas  # Assurez-vous d'avoir cette bibliothèque installée si vous souhaitez utiliser hvplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from utils import print_evaluate

 

def process_linear_regression(planet_type, df_planet, Echantillion_min):
    """
    Fonction pour effectuer une régression linéaire et afficher les résultats pour un type de planète donné.
    
    Parameters:
    - planet_type: Le type de planète à traiter.
    - df_planet: Le DataFrame contenant les données des planètes.
    """
    print(f"Traitement pour le type de planète : {planet_type}")
    
    # Filtrer les données pour ce type de planète
    type_planet_df = df_planet[df_planet['TypeCoreName'] == planet_type]
    
    if len(type_planet_df) > Echantillion_min:  # Exclure les types de planètes avec moins de 100 valeurs
        # Sélectionner les colonnes 'radius' et 'density'
        SystemPlaneteEchantillon = type_planet_df[['radius', 'density']]
        
        # Variables indépendantes et dépendantes
        X = SystemPlaneteEchantillon[['radius']]  # Variable indépendante (radius)
        y = SystemPlaneteEchantillon['density']   # Variable dépendante (density)
        
        # Gérer les valeurs manquantes
        X = X.fillna(X.mean(numeric_only=True))
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Pipeline de normalisation
        PipelineCal = Pipeline([('std_scaler', StandardScaler())])  
        X_train = PipelineCal.fit_transform(X_train)
        X_test = PipelineCal.transform(X_test)
        
        # Régression linéaire
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, Y_train)
        
        # Coefficients du modèle
        print(f"Intercept pour {planet_type} : ", lin_reg.intercept_)
        coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
        print(coeff_df)
        
        # Prédictions
        pred = lin_reg.predict(X_test)
        
        # Visualisation
        scatter_plot = pd.DataFrame({'True Values': Y_test, 'Predicted Values': pred}).hvplot.scatter(x='True Values', y='Predicted Values', title=f"True vs Predicted - {planet_type}")
        error_plot = pd.DataFrame({'Error Values': (Y_test - pred)}).hvplot.kde(title=f"Error Distribution - {planet_type}")
        
        # Affichage du Grapiques
        scatter_plot + error_plot
        
        # Optionnel : Utilisation de Matplotlib pour plus de personnalisation
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, pred, color='blue', alpha=0.6)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)  # Ligne de référence
        plt.xlabel("True Density")
        plt.ylabel("Predicted Density")
        plt.title(f"True vs Predicted - Linear Regression - {planet_type}")
        plt.grid(True)
        plt.show()

        # Plot de la distribution des erreurs
        plt.figure(figsize=(10, 6))
        plt.hist(Y_test - pred, bins=30, color='red', alpha=0.7)
        plt.title(f"Error Distribution - {planet_type}")
        plt.xlabel("Error (True - Predicted)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
        # Evaluation des résultats
        start_time = time.time()
        test_pred = lin_reg.predict(X_test)
        train_pred = lin_reg.predict(X_train)
        
        print(f"Évaluation pour le type de planète {planet_type}:")
        print_evaluate(Y_test, test_pred, 'Test set evaluation:')
        print_evaluate(Y_train, train_pred, 'Train set evaluation:')
        
        end_time = time.time()

        # Résultat dans un dataframe
        results_df = pd.DataFrame(data=[["Linear Regression", mean_absolute_error(Y_test, test_pred), mean_squared_error(Y_test, test_pred), 
                                        mean_squared_error(Y_test, test_pred, squared=False), r2_score(Y_test, test_pred), 
                                        end_time - start_time]], 
                                  columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Execution Time'])

        print(results_df)
        print("\n" + "-"*50 + "\n")
    else:
        print(f"Le type de planète {planet_type} contient trop peu d'échantillons pour une analyse (moins de 100 échantillons).")

