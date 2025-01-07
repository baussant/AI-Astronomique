import numpy as np # type: ignore
import pandas as pd # type: ignore
import warnings
import hvplot.pandas
import matplotlib.pyplot as plt # type: ignore
import tensorflow as tf # type: ignore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler,label_binarize
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay,roc_auc_score,roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.initializers import HeNormal # type: ignore
from imblearn.over_sampling import SMOTE
from IPython.display import display, HTML # Pour afficher les données dans Jupyter Notebook avec un format HTML
# ***********************   Librairie personnelle  **************************************
from utils import process_and_plot_graphs2,print_evaluate,evaluate,prepare_data_graph
from EcritureCSV import create_csv
from DataProcessing import Nbr_Moons2,get_unique_planet_types
from ToolsAI import Calcul_XX_YY_2
from Evetools import TriData
from ModelHistory import save_dp_results

def Deep(Smote_State,df_planet,Echantillon_min,Y_Target,X_Chara,Weight_Class,Adjust_Factor,Epoque,batch_size_nbr,Learning_Rate,Save_Model):

    # Charger les données
    df = df_planet.copy()

    # Filtrer les classes avec suffisamment de données
    class_counts = df['TypeCoreName'].value_counts()
    valid_classes = class_counts[class_counts > Echantillon_min].index
    data = df[df['TypeCoreName'].isin(valid_classes)]

    # Calcul des données d'entrée et de sortie
    X, y, encoder = Calcul_XX_YY_2(data, Y_Target, X_Chara)

    # Conversion des étiquettes en one-hot
    y_categorical = to_categorical(y, num_classes=len(encoder.classes_))

    #print("Labels encodés (premiers exemples) :", y[:5])
    #print("Labels encodés (catégoriques) :", y_categorical[:5])

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer SMOTE pour équilibrer les classes
    if Smote_State:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_categorical)  

    else:
        X_resampled, y_resampled = X_scaled,y_categorical 
   
    # Diviser en ensemble d'entraînement, de validation et de test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Reshape des données pour les couches convolutives
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Si y_resampled est déjà en one-hot, le convertir en indices de classes
    if len(y_resampled.shape) > 1:
        y_resampled = np.argmax(y_resampled, axis=1)
    classes = np.unique(y_resampled)
    

    # Compute class weights
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_resampled)
    class_weights = dict(enumerate(class_weights))

    if Weight_Class == True:
        # Affichage des poids initiaux
        print("Poids des classes avant ajustement :", class_weights)

        # Augmenter les poids pour les classes avec faible rappel
        adjustment_factor = Adjust_Factor  # Facteur d'augmentation des poids pour les classes faibles
        class_weights[0] *= adjustment_factor  # Augmenter le poids pour Planet (Barren)
        class_weights[4] *= adjustment_factor  # Augmenter le poids pour Planet (Oceanic)

        # Affichage des poids ajustés
        print("Poids des classes après ajustement :", class_weights)

    # Nouveau modèle avec couches convolutives 1D
    model = Sequential([
        # Première couche convolutive 1D
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Deuxième couche convolutive 1D
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=1),
        Dropout(0.4),

        # Aplatissement des données pour passer aux couches denses
        Flatten(),

        # Première couche dense avec L2
        Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        BatchNormalization(),
        Dropout(0.4),

        # Deuxième couche dense avec L2
        Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        BatchNormalization(),
        Dropout(0.4),

        # Troisième couche dense
        Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),

        # Couche de sortie
        Dense(len(encoder.classes_), activation='softmax')
    ])

    # Compilation du modèle avec un optimiseur Adam
    optimizer = Adam(learning_rate=Learning_Rate)  # Taux d'apprentissage ajusté
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Vérifier si TensorFlow détecte un GPU
    print("GPU disponible : ", tf.config.list_physical_devices('GPU'))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Callbacks pour éviter le surentraînement et ajuster dynamiquement le taux d'apprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-5)

    # Entraîner le modèle
    lr_logger = LearningRateLogger()
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=Epoque,
        batch_size=batch_size_nbr,
        class_weight=class_weights,  # Application des poids de classe
        callbacks=[early_stopping, reduce_lr, lr_logger])

    images, accuracy, accuracy_train,report = Affichage(model,y_test,X_test_cnn,encoder,X_train_cnn,y_train,history,lr_logger)

    if Save_Model:
        save_dp_results(Smote_State,report,model,scaler, accuracy, accuracy_train, images,Echantillon_min,Weight_Class,Adjust_Factor,Epoque,batch_size_nbr,Learning_Rate)
        #   model.save('model_planet_classification_cnn.h5')


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        if not hasattr(self, 'lr_history'):
            self.lr_history = []
        self.lr_history.append(lr)


def Affichage(model,y_test,X_test_cnn,encoder,X_train_cnn,y_train,history,lr_logger):

    # Évaluation
    loss_test, accuracy_test = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"-------------------------------------------------------\n")
    print(f"Précision sur les données de test : {accuracy_test:.2f}\n")

    # Évaluation sur les données d'entraînement
    loss_train, accuracy_train = model.evaluate(X_train_cnn, y_train, verbose=0)
    print(f"-------------------------------------------------------\n")
    print(f"Précision sur les données d'entraînement : {accuracy_train:.2f}\n")

    # Générer un rapport de classification détaillé
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test_classes = y_test.argmax(axis=1)
    report = classification_report(y_test_classes, y_pred_classes, target_names=encoder.classes_)
    print(f"-------------------------------------------------------\n")
    print(f"---------------- Rapport sur données de Test ----------\n")
    print(f"-------------------------------------------------------\n")
    print(report)

    Y_pred_train = model.predict(X_train_cnn)
    Y_pred_train_classes = Y_pred_train.argmax(axis=1)
    Y_train_classes = y_train.argmax(axis=1)
    report_train = classification_report(Y_train_classes, Y_pred_train_classes, target_names=encoder.classes_)
    print(f"-------------------------------------------------------\n")
    print(f"---------------- Rapport sur données de Train ---------\n")
    print(f"-------------------------------------------------------\n")
    print(report_train)

    # Tracer les courbes de perte (loss)
    plt.figure()
    plt.plot(history.history['loss'], label='Loss - Entraînement')
    plt.plot(history.history['val_loss'], label='Loss - Validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.title('Perte en fonction des époques')
    plt.legend()
    fig_loss = plt.gcf()
    plt.show()

    # Tracer les courbes de précision (accuracy)
    plt.figure()
    plt.plot(history.history['accuracy'], label='Précision - Entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision - Validation')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.title('Précision en fonction des époques')
    plt.legend()
    fig_accuracy = plt.gcf()
    plt.show()

    y_test_binary = label_binarize(y_test_classes, classes=range(len(encoder.classes_)))
    y_pred_prob = model.predict(X_test_cnn)

    for i, class_name in enumerate(encoder.classes_):
        fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonale
    plt.title("Courbe ROC par Classe")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    fig_roc = plt.gcf()
    plt.show()

    plt.plot(range(len(lr_logger.lr_history)), lr_logger.lr_history)
    plt.title("Évolution du Taux d'Apprentissage")
    plt.xlabel("Époques")
    plt.ylabel("Taux d'Apprentissage")
    fig_apprentissage = plt.gcf()
    plt.show()

    # Générer et afficher une matrice de confusion
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_).plot(cmap='viridis')
    plt.title('Matrice de confusion')
    plt.xticks(rotation=90)
    fig_confusion = plt.gcf()
    plt.show()

    images=[]    
    images.append(("Confusion Matrix", fig_confusion))
    images.append(("Tracer les courbes de précision (accuracy)", fig_accuracy))
    images.append(("Tracer les courbes de perte (loss)", fig_loss)) 
    images.append(("Tracer les courbes ROC et AUC ", fig_roc)) 
    images.append(("Tracer les courbes de l'évolution du taux d'apprentissage ", fig_apprentissage)) 

    return images, accuracy_test, accuracy_train,report
