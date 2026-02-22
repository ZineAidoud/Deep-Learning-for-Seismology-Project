# Détection automatique d'événements sismiques par Deep Learning

Ce projet vise à automatiser la détection d'événements sismiques majeurs à l'aide de réseaux de neurones profonds. Réalisé dans le cadre du Master 2 Ingénierie Mathématique et Data Science à l'Université de Haute-Alsace, l'objectif est de fournir un outil d'aide à la décision robuste et léger en ressources pour les sismologues.

# Contexte et Données
* **Dataset** : Données `Earthquakes` issues de l'archive UCR (collectées par le Northern California Earthquake Data Center).
* **Format** :Séries temporelles univariées composées de 512 points de mesure.
* **Défi technique** : Fort déséquilibre des classes (*Class Imbalance*). La classe événement majeur est fortement sous-représentée.

## Architectures

Le notebook `Projet_final_apprentissage__profond.ipynb` contient l'implémentation et la comparaison de trois architectures sous contrainte de ressources limitées :

### 1. Multi-Layer Perceptron (MLP)
* **Approche** : Remplacement des données brutes par un *Feature Engineering* intensif (extraction de 9 descripteurs physiques tels que le Kurtosis, l'énergie FFT et le Zero Crossing Rate).
* **Paramètres** : 2800 paramètres.
* **Résultat** : Détection viable avec un seuil de décision abaissé à 0.35, priorisant le rappel (*Recall*).

### 2. Réseau de Neurones Convolutif (CNN 1D) - Modèle recommandé
* **Approche** : Apprentissage direct sur les séquences brutes. Utilisation d'une couche *Global Average Pooling* (GAP) pour réduire drastiquement la dimensionnalité.
* **Paramètres** : 30000 paramètres (contre plus de 200 000 pour un CNN standard avec *Flatten*).
* **Résultat** : Modèle offrant le meilleur filet de sécurité avec un **Rappel de 74%** sur la classe des séismes. 

### 3. Réseau Récurrent (SimpleRNN)
* **Approche** : Traitement séquentiel avec *Gradient Clipping* (clipnorm=1.0) pour capturer les variations progressives d'amplitude.
* **Paramètres** : 25000 paramètres.
* **Résultat** : Bonne efficacité computationnelle mais souffre d'un rappel insuffisant (40%) sur la classe critique dans un contexte de forte asymétrie.

## Méthodologie d'optimisation
Pour contrer le déséquilibre des classes, l'intégration de poids de classes (`class_weight`) automatisés a été indispensable. Des techniques de régularisation avancées (*SpatialDropout1D*, *Early Stopping*, *ReduceLROnPlateau*) ont également été déployées pour stabiliser l'apprentissage.


## Auteurs
* Hocine Rayane ARHAB
* Zine Elabidine AIDOUD
* Sidy DIOP
