# API de Prédiction de Maintenance de Véhicules

API FastAPI utilisant un modèle de Machine Learning pour prédire le moment optimal pour la prochaine maintenance d'un véhicule.

## Description

Cette API permet de prédire le nombre de jours restants avant la prochaine maintenance recommandée d'un véhicule en fonction de multiples paramètres (kilométrage, historique des révisions, condition du véhicule, etc.).

Le modèle utilise un algorithme de régression linéaire multiple (scikit-learn) entraîné sur un dataset de 5000 véhicules avec leurs historiques de maintenance.

## Fonctionnalités

- Prédiction du nombre de jours avant la prochaine maintenance
- Calcul d'une fourchette de confiance (intervalle min/max)
- Recommandations personnalisées basées sur l'urgence
- API REST simple et documentée automatiquement (Swagger UI)

## Prérequis

- Python 3.12
- pip
- Docker (optionnel)

## Installation

### Installation locale

1. Cloner le projet

```bash
git clone https://github.com/slayercode1/ia_cmv
cd ia_cmv
```

2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### Installation avec Docker

```bash
docker build -t ia-cmv-api .
docker run -p 8000:8000 ia-cmv-api
```

## Utilisation

### Entraîner le modèle (optionnel)

Si vous souhaitez ré-entraîner le modèle ou expérimenter avec le dataset:

1. Ouvrez le notebook [Regression_Lineaire.ipynb](Regression_Lineaire.ipynb) dans Google Colab
2. Uploadez le fichier [dataset_vehicules_5000.csv](dataset_vehicules_5000.csv) dans l'environnement Colab
3. Exécutez toutes les cellules du notebook
4. Téléchargez le nouveau fichier `model_multiple2.pkl` généré
5. Remplacez le modèle dans le projet

### Lancer l'API

**En local:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Avec Docker:**

```bash
docker run -p 8000:8000 ia-cmv-api
```

### Accéder à la documentation

Une fois l'API lancée, accédez à:

- Documentation Swagger UI: <http://localhost:8000/docs>
- Documentation ReDoc: <http://localhost:8000/redoc>

### Exemple de requête

**Endpoint:** `POST /predict`

**Corps de la requête (JSON):**

```json
{
  "km_actuel": 45000,
  "km_moyen_annuel": 15000,
  "km_derniere_revision": 42000,
  "jours_depuis_derniere_revision": 90,
  "km_depuis_derniere_revision": 3000,
  "nb_revisions_effectuees": 3,
  "intervalle_recommande_jours": 365,
  "intervalle_recommande_km": 15000,
  "condition_vehicule": 8,
  "nb_pannes_historique": 1,
  "age_vehicule": 3,
  "taux_utilisation_km": 0.2,
  "taux_utilisation_jours": 0.25,
  "revisions_par_an": 1.0,
  "Carburant_Factor": 1.2,
  "Usage_Factor": 0.8
}
```

**Exemple avec curl:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data_test.json
```

**Réponse:**

```json
{
  "estimation_jours": 180,
  "estimation_mois": 6.0,
  "fourchette_min_mois": 5.0,
  "fourchette_max_mois": 7.0,
  "recommandation": "Planifier la maintenance dans 5 à 9 mois"
}
```

## Paramètres d'entrée

| Paramètre                        | Type  | Description                                        |
| -------------------------------- | ----- | -------------------------------------------------- |
| `km_actuel`                      | float | Kilométrage actuel du véhicule                     |
| `km_moyen_annuel`                | float | Kilométrage moyen annuel                           |
| `km_derniere_revision`           | float | Kilométrage lors de la dernière révision           |
| `jours_depuis_derniere_revision` | int   | Nombre de jours depuis la dernière révision        |
| `km_depuis_derniere_revision`    | float | Kilométres parcourus depuis la dernière révision   |
| `nb_revisions_effectuees`        | int   | Nombre total de révisions effectuées               |
| `intervalle_recommande_jours`    | int   | Intervalle recommandé en jours par le constructeur |
| `intervalle_recommande_km`       | float | Intervalle recommandé en km par le constructeur    |
| `condition_vehicule`             | int   | État du véhicule (échelle 1-10)                    |
| `nb_pannes_historique`           | int   | Nombre de pannes dans l'historique                 |
| `age_vehicule`                   | int   | Âge du véhicule en années                          |
| `taux_utilisation_km`            | float | Taux d'utilisation basé sur les km                 |
| `taux_utilisation_jours`         | float | Taux d'utilisation basé sur les jours              |
| `revisions_par_an`               | float | Nombre moyen de révisions par an                   |
| `Carburant_Factor`               | float | Facteur lié au type de carburant                   |
| `Usage_Factor`                   | float | Facteur lié au type d'usage                        |

## Structure du projet

```
ia_cmv/
├── main.py                        # Application FastAPI principale
├── model_multiple2.pkl            # Modèle ML entraîné
├── Regression_Lineaire.ipynb      # Notebook Colab d'entraînement du modèle
├── dataset_vehicules_5000.csv     # Dataset d'entraînement (5000 véhicules)
├── requirements.txt               # Dépendances Python
├── Dockerfile                     # Configuration Docker
├── .dockerignore                  # Fichiers à exclure du build Docker
├── data_test.json                 # Données de test exemple
└── README.md                      # Documentation
```

## Modèle de Machine Learning

### Entraînement du modèle

Le modèle a été entraîné dans le notebook [Regression_Lineaire.ipynb](Regression_Lineaire.ipynb) disponible dans le projet. Ce notebook peut être exécuté sur Google Colab pour:

- Explorer le dataset de 5000 véhicules
- Analyser les corrélations entre les features
- Entraîner différents modèles de régression
- Évaluer les performances (RMSE, R², MAE)
- Générer le fichier `model_multiple2.pkl`

### Dataset

Le fichier [dataset_vehicules_5000.csv](dataset_vehicules_5000.csv) contient 5000 enregistrements de véhicules avec:

- Données de kilométrage et d'utilisation
- Historique des révisions
- Informations sur l'état du véhicule
- Type de carburant et d'usage
- Jours avant la prochaine maintenance (variable cible)

### Modèle sauvegardé

Le modèle est sauvegardé dans [model_multiple2.pkl](model_multiple2.pkl) et contient:

- Le modèle de régression linéaire multiple entraîné
- Le scaler StandardScaler pour la normalisation des features
- Le RMSE (Root Mean Square Error) pour calculer l'intervalle de confiance

## Catégories de recommandations

- **< 60 jours**: Maintenance très urgente (< 2 mois)
- **60-150 jours**: Maintenance à planifier dans 2-5 mois
- **150-270 jours**: Maintenance à planifier dans 5-9 mois
- **> 270 jours**: Maintenance non urgente (> 9 mois)

## Dépendances

- FastAPI 0.124.2
- Uvicorn 0.38.0
- Pydantic 2.10.6
- scikit-learn 1.6.1
- numpy 2.3.5

## Auteur

Yann Clain
