import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,roc_curve,auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
import seaborn as sns # type: ignore

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

def Affichage_base(x_train, x_test, encoder, PCA_Number, best_k, grid_search):
    print(f"-----------------------------------------------------------------------------------------\n")
    print(f"Entraînement sur {x_train.shape[0]} échantillons, test sur {x_test.shape[0]} échantillons")
    print(f"Nombre de classes : {len(encoder.classes_)}")
    print(f"Classes : {encoder.classes_}")
    print(f"Nombre de composantes principales : {PCA_Number}")
    print(f"Nombre de voisins : {best_k}")
    print(f"Meilleurs paramètres : {grid_search.best_estimator_}")
    print(f"-----------------------------------------------------------------------------------------\n")

def Affichage_confusion_matrix(y_test, y_pred, encoder):
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables')
    plt.title('Matrice de confusion')
    fig_confusion = plt.gcf()
    plt.show()
    return fig_confusion

def Affichage_roc_curve(y_test, y_pred, encoder, best_model, x_test):
    # Courbe ROC
    pred_probs = best_model.predict_proba(x_test)
    y_test_binarized = label_binarize(y_test, classes=range(len(encoder.classes_)))
    n_classes = y_test_binarized.shape[1]
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
    return fig_roc

def Affichage_roc_curve2(y_test, y_pred, encoder):
    # Courbe ROC

    y_test_bin = encoder.transform(y_test)
    y_pred_bin = encoder.transform(y_pred)
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_bin)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    fig_roc = plt.gcf()
    plt.show()
    return fig_roc

def Affichage_graphique(X_train, y_train, encoder):
    # Visualisation des données
    df = pd.DataFrame(X_train, columns=['X1', 'X2'])
    df['y'] = encoder.inverse_transform(y_train)
    sns.lmplot(x='X1', y='X2', data=df, hue='y', fit_reg=False, height=7)
    plt.title('Visualisation des données')
    fig_data = plt.gcf()
    plt.show()
    return fig_data

def Affichage_graphique_3D(X_train, y_train, encoder):
    # Visualisation des données en 3D
    df = pd.DataFrame(X_train, columns=['X1', 'X2', 'X3'])
    df['y'] = encoder.inverse_transform(y_train)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(encoder.classes_, colors):
        indicesToKeep = df['y'] == target
        ax.scatter(df.loc[indicesToKeep, 'X1'], df.loc[indicesToKeep, 'X2'], df.loc[indicesToKeep, 'X3'], c=color, s=50)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Visualisation des données en 3D')
    fig_data_3D = plt.gcf()
    plt.show()
    return fig_data_3D

def Affichage_courbe_apprentissage(best_model, X_train, Y_train, cv_value):
    train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, Y_train, cv=cv_value, scoring='accuracy')
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
    return fig_CurveLearning

def Affichage_courbe_apprentissage2(history):
    # Courbes d'apprentissage
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Courbes d\'apprentissage')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig_learning_curve = plt.gcf()
    plt.show()
    return fig_learning_curve

def Affichage_courbe_perte(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Courbes de perte')
    plt.ylabel('Perte')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig_loss_curve = plt.gcf()
    plt.show()
    return fig_loss_curve

def Affichage_courbe_precision(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Courbes de précision')
    plt.ylabel('Précision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig_precision_curve = plt.gcf()
    plt.show()
    return fig_precision_curve

def Affichage_courbe_recall(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Courbes de rappel')
    plt.ylabel('Rappel')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig_recall_curve = plt.gcf()
    plt.show()
    return fig_recall_curve

def Affichage_courbe_f1(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('Courbes de F1')
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig_f1_curve = plt.gcf()
    plt.show()
    return fig_f1_curve

def Affichage_proba(y_test_proba):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test_proba.max(axis=1), bins=20, kde=True, color='blue')
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Max Probability")
    plt.ylabel("Frequency")
    fig_proba = plt.gcf()
    plt.show()
    return fig_proba

def Affichage_proba2(Y_test_proba,valid_classes):
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(valid_classes):
        sns.kdeplot(Y_test_proba[:, i], label=f'Class {class_name}')
    plt.title('Probability Distributions by Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    fig_Predict = plt.gcf()
    plt.show()
    return fig_Predict