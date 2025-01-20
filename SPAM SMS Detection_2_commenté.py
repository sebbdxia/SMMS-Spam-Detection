
# Importation des bibliothèques nécessaires
import streamlit as st  # Framework pour créer des applications web interactives
import pandas as pd  # Manipulation et analyse des données
import numpy as np  # Opérations mathématiques et manipulation de tableaux
import matplotlib.pyplot as plt  # Génération de graphiques statiques
from sklearn.feature_extraction.text import TfidfVectorizer  # Conversion de texte en vecteurs numériques
from sklearn.model_selection import train_test_split  # Division des données en ensembles d'entraînement et de test
from sklearn.linear_model import LogisticRegression  # Modèle de régression logistique
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve  # Évaluation des modèles
from fpdf import FPDF  # Création de fichiers PDF
import seaborn as sns  # Visualisation des données (complément à matplotlib)
import os  # Gestion des chemins et fichiers

# Initialisation des variables globales
model = None  # Stockage du modèle entraîné
vectorizer = None  # Stockage du vectoriseur TF-IDF
combined_data = None  # Stockage des données fusionnées
threshold = 0.3  # Seuil pour classifier un message comme SPAM ou HAM

# Fonction pour nettoyer les messages
def clean_message(message):
    # Liste des mots vides (stopwords) à supprimer
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])
    # Conversion en minuscules, suppression des caractères non alphanumériques
    message = ''.join([char for char in message.lower() if char.isalnum() or char.isspace()])
    # Suppression des stopwords et renvoi du texte nettoyé
    return ' '.join([word for word in message.split() if word not in stopwords])

# Fonction pour charger les données depuis un fichier CSV
def load_data(file):
    # Lecture des données avec pandas (colonnes : label et message)
    data = pd.read_csv(file, sep='\t', header=None, names=['label', 'message'])
    # Application du nettoyage des messages
    data['cleaned_message'] = data['message'].apply(clean_message)
    return data

# Fonction pour fusionner et nettoyer deux jeux de données
def merge_and_clean_data(original, new):
    # Nettoyage de la nouvelle base de données
    new['cleaned_message'] = new['message'].apply(clean_message)
    # Fusion des bases de données et suppression des doublons
    combined = pd.concat([original, new], ignore_index=True).drop_duplicates()
    return combined

# Fonction pour entraîner le modèle
def train_model(data):
    global model, vectorizer  # Utilisation des variables globales
    # Conversion des messages nettoyés en vecteurs numériques (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_message'])
    # Conversion des labels en valeurs binaires (spam : 1, ham : 0)
    y = (data['label'] == 'spam').astype(int)
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Initialisation du modèle de régression logistique
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    # Entraînement du modèle
    model.fit(X_train, y_train)
    # Prédictions des probabilités sur l'ensemble de test
    y_probs = model.predict_proba(X_test)[:, 1]
    # Prédictions finales avec le seuil ajusté
    y_pred = (y_probs >= threshold).astype(int)
    # Calcul des métriques de classification et de la matrice de confusion
    metrics = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return metrics, conf_matrix, y_probs, y_test

# Fonction pour afficher des graphiques dynamiques
def display_graphs(data, conf_matrix, y_probs, y_test):
    st.subheader("Visualisation des Résultats")
    # Distribution des classes (HAM vs SPAM)
    st.write("Distribution des classes")
    class_distribution = data['label'].value_counts()
    fig, ax = plt.subplots()
    class_distribution.plot(kind='bar', color=['blue', 'orange'], ax=ax)
    plt.title("Distribution des classes (HAM vs SPAM)")
    plt.xlabel("Classe")
    plt.ylabel("Nombre de messages")
    st.pyplot(fig)

    # Matrice de confusion
    st.write("Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["HAM", "SPAM"], yticklabels=["HAM", "SPAM"], ax=ax)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Classe réelle")
    st.pyplot(fig)

    # Courbe précision-rappel
    st.write("Courbe Précision-Rappel")
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Précision-Rappel")
    plt.title("Courbe Précision-Rappel")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    st.pyplot(fig)

# Interface utilisateur Streamlit
st.title("Application de Détection de SPAM SMS")
st.sidebar.title("Options")

# Chargement initial de la base de données
default_data_path = "C:\\Users\\sbond\\Desktop\\SPAM Sms detection\\SMSSpamCollection.csv"
original_data = load_data(default_data_path)
combined_data = original_data.copy()

# Entraînement initial du modèle
metrics, conf_matrix, y_probs, y_test = train_model(combined_data)

# Section : Prédiction d'un SMS unique
st.sidebar.subheader("Tester un SMS")
input_sms = st.sidebar.text_input("Entrez le SMS à tester")
if st.sidebar.button("Prédire"):
    if model is not None and vectorizer is not None:
        cleaned_sms = clean_message(input_sms)
        vectorized_sms = vectorizer.transform([cleaned_sms])
        prob = model.predict_proba(vectorized_sms)[0, 1]
        prediction = "SPAM" if prob >= threshold else "HAM"
        st.write(f"Le SMS est classifié comme : **{prediction}** avec une probabilité de SPAM de {prob:.2f}")

# Section : Téléchargement d'une nouvelle base et fusion
st.sidebar.subheader("Télécharger une nouvelle base de données")
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier CSV", type="csv")
if uploaded_file is not None:
    new_data = load_data(uploaded_file)
    st.write("Nouvelle base de données chargée avec succès !")
    combined_data = merge_and_clean_data(combined_data, new_data)
    st.write("Base de données fusionnée et nettoyée.")
    metrics, conf_matrix, y_probs, y_test = train_model(combined_data)
    st.write("Analyse des données fusionnées complétée.")
    display_graphs(combined_data, conf_matrix, y_probs, y_test)

# Section : Réentraînement avec données fusionnées
if st.sidebar.button("Réentraîner avec les données fusionnées"):
    metrics, conf_matrix, y_probs, y_test = train_model(combined_data)
    st.write("Le modèle a été réentraîné avec succès.")
    display_graphs(combined_data, conf_matrix, y_probs, y_test)
