import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from fpdf import FPDF
import seaborn as sns
import os

# Initialisation des variables globales
model = None
vectorizer = None
combined_data = None
threshold = 0.3  # Seuil ajusté pour augmenter la sensibilité aux SPAM

# Nettoyage des messages
def clean_message(message):
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
    message = ''.join([char for char in message.lower() if char.isalnum() or char.isspace()])
    return ' '.join([word for word in message.split() if word not in stopwords])

# Charger les données
def load_data(file):
    data = pd.read_csv(file, sep='\t', header=None, names=['label', 'message'])
    data['cleaned_message'] = data['message'].apply(clean_message)
    return data

# Fusionner et uniformiser les bases de données
def merge_and_clean_data(original, new):
    new['cleaned_message'] = new['message'].apply(clean_message)
    combined = pd.concat([original, new], ignore_index=True).drop_duplicates()
    return combined

# Entraîner le modèle
def train_model(data):
    global model, vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['cleaned_message'])
    y = (data['label'] == 'spam').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return metrics, conf_matrix, y_probs, y_test

# Afficher les graphiques dynamiques
def display_graphs(data, conf_matrix, y_probs, y_test):
    st.subheader("Visualisation des Résultats")

    # Distribution des classes
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

# Interface utilisateur Streamlit
st.title("Application de Détection de SPAM SMS")
st.sidebar.title("Options")

# Charger la base de données initiale
default_data_path = "C:\\Users\\sbond\\Desktop\\SPAM Sms detection\\SMSSpamCollection.csv"
original_data = load_data(default_data_path)
combined_data = original_data.copy()

# Entraîner le modèle sur les données initiales
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
