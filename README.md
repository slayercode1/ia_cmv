# API de Prediction de Maintenance de Vehicules

API FastAPI utilisant un modele XGBoost pour predire le moment optimal pour la prochaine maintenance d'un vehicule.

## Description

Cette API predit le nombre de jours restants avant la prochaine maintenance recommandee d'un vehicule en fonction de 16 parametres (kilometrage, historique des revisions, condition du vehicule, etc.).

Le modele utilise un **Pipeline** scikit-learn (StandardScaler + XGBoost) optimise avec **GridSearchCV**, entraine sur un dataset de 5000 vehicules. Il est versionne et servi via **MLflow Model Registry**.

## Fonctionnalites

- Prediction du nombre de jours avant la prochaine maintenance
- Calcul d'une fourchette de confiance (intervalle min/max)
- Recommandations personnalisees basees sur l'urgence
- API REST documentee automatiquement (Swagger UI / ReDoc)
- Versioning du modele via MLflow Model Registry
- Rechargement du modele a chaud (`POST /reload-model`)

## Prerequis

- Python 3.12
- Docker et Docker Compose

## Lancement avec Docker Compose (recommande)

Docker Compose demarre deux services : le serveur **MLflow** (tracking + registry) et l'**API FastAPI**.

```bash
docker compose up --build -d
```

| Service | URL | Description |
| ------- | --- | ----------- |
| API     | http://localhost:8000 | API de prediction |
| Swagger | http://localhost:8000/docs | Documentation interactive |
| ReDoc   | http://localhost:8000/redoc | Documentation alternative |
| MLflow  | http://localhost:5000 | UI MLflow (experiments, registry) |

### Entrainer le modele et l'enregistrer dans MLflow

1. S'assurer que le serveur MLflow tourne (`docker compose up -d mlflow`)
2. Ouvrir le notebook `notebook/Regression_Lineaire.ipynb` avec Jupyter
3. Executer toutes les cellules : le modele est automatiquement enregistre dans MLflow avec l'alias `champion`
4. Recharger le modele dans l'API :

```bash
curl -X POST http://localhost:8000/reload-model
```

### Arreter les services

```bash
docker compose down       # Garde les donnees MLflow
docker compose down -v    # Supprime aussi le volume MLflow
```

## Installation locale (developpement)

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

L'API necessite un serveur MLflow avec un modele enregistre pour fonctionner. Sans modele, l'endpoint `/predict` retourne une erreur 503.

## Utilisation de l'API

### Exemple de requete

**Endpoint :** `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "km_actuel": 100000,
    "km_moyen_annuel": 15000,
    "km_derniere_revision": 95000,
    "jours_depuis_derniere_revision": 150,
    "km_depuis_derniere_revision": 5000,
    "nb_revisions_effectuees": 10,
    "intervalle_recommande_jours": 365,
    "intervalle_recommande_km": 10000,
    "condition_vehicule": 8,
    "nb_pannes_historique": 1,
    "age_vehicule": 7,
    "taux_utilisation_km": 1.2,
    "taux_utilisation_jours": 1.0,
    "revisions_par_an": 1.5,
    "Carburant_Factor": 1.0,
    "Usage_Factor": 1.2
  }'
```

**Reponse :**

```json
{
  "estimation_jours": 157,
  "estimation_mois": 5.2,
  "fourchette_min_mois": 4.3,
  "fourchette_max_mois": 6.2,
  "recommandation": "Planifier la maintenance dans 5 a 9 mois"
}
```

## Parametres d'entree

| Parametre                        | Type  | Description                                        |
| -------------------------------- | ----- | -------------------------------------------------- |
| `km_actuel`                      | float | Kilometrage actuel du vehicule                     |
| `km_moyen_annuel`                | float | Kilometrage moyen annuel                           |
| `km_derniere_revision`           | float | Kilometrage lors de la derniere revision           |
| `jours_depuis_derniere_revision` | int   | Nombre de jours depuis la derniere revision        |
| `km_depuis_derniere_revision`    | float | Kilometres parcourus depuis la derniere revision   |
| `nb_revisions_effectuees`        | int   | Nombre total de revisions effectuees               |
| `intervalle_recommande_jours`    | int   | Intervalle recommande en jours par le constructeur |
| `intervalle_recommande_km`       | float | Intervalle recommande en km par le constructeur    |
| `condition_vehicule`             | int   | Etat du vehicule (echelle 1-10)                    |
| `nb_pannes_historique`           | int   | Nombre de pannes dans l'historique                 |
| `age_vehicule`                   | int   | Age du vehicule en annees                          |
| `taux_utilisation_km`            | float | Taux d'utilisation base sur les km                 |
| `taux_utilisation_jours`         | float | Taux d'utilisation base sur les jours              |
| `revisions_par_an`               | float | Nombre moyen de revisions par an                   |
| `Carburant_Factor`               | float | Facteur carburant (Essence=1.0, Diesel=1.1, Hybride=1.2, Electrique=1.5) |
| `Usage_Factor`                   | float | Facteur usage (Flotte=0.8, Professionnel=0.9, Personnel=1.2) |

## Categories de recommandations

| Estimation       | Recommandation                            |
| ---------------- | ----------------------------------------- |
| < 60 jours       | Maintenance tres urgente (< 2 mois)      |
| 60 - 150 jours   | Planifier dans 2 a 5 mois                |
| 150 - 270 jours  | Planifier dans 5 a 9 mois                |
| > 270 jours      | Pas urgent (> 9 mois)                    |

## Structure du projet

```
ia_cmv/
├── app/
│   ├── main.py                    # Application FastAPI
│   └── test_main.py               # Tests (pytest)
├── notebook/
│   └── Regression_Lineaire.ipynb  # Entrainement + enregistrement MLflow
├── dataset_vehicules_5000.csv     # Dataset (5000 vehicules)
├── data_test.json                 # Donnees de test exemple
├── requirements.txt               # Dependances Python
├── Dockerfile                     # Image Docker de l'API
├── Dockerfile.mlflow              # Image Docker du serveur MLflow
├── docker-compose.yml             # Orchestration API + MLflow
├── .dockerignore                  # Fichiers exclus du build Docker
├── .gitignore                     # Fichiers exclus de git
└── README.md                      # Documentation
```

## Deploiement sur un VPS

### Prerequis sur le serveur

- Un VPS avec Docker et Docker Compose installes
- Un nom de domaine pointant vers l'IP du VPS (pour HTTPS)

### 1. Deployer le projet

```bash
# Cloner le depot
git clone https://github.com/slayercode1/ia_cmv.git
cd ia_cmv

# Lancer les conteneurs en arriere-plan
docker compose up --build -d
```

L'API tourne maintenant sur le port 8000 et MLflow sur le port 5000.

### 2. Entrainer et charger le modele

Depuis votre machine locale (avec le notebook Jupyter) :

```bash
# Le notebook doit pointer vers le MLflow du VPS
export MLFLOW_TRACKING_URI=http://votre-ip:5000
```

Modifier la variable `MLFLOW_TRACKING_URI` dans la premiere cellule du notebook :

```python
MLFLOW_TRACKING_URI = "http://votre-ip:5000"
```

Puis executer le notebook. Une fois termine, recharger le modele dans l'API :

```bash
curl -X POST http://votre-ip:8000/reload-model
```

### 3. Configurer Caddy (HTTPS automatique)

Caddy gere automatiquement les certificats SSL via Let's Encrypt.

Installer Caddy :

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

Editer le fichier `/etc/caddy/Caddyfile` :

```
api.votre-domaine.com {
    reverse_proxy 127.0.0.1:8000
}
```

Relancer Caddy :

```bash
sudo systemctl reload caddy
```

L'API est maintenant accessible en HTTPS sur `https://api.votre-domaine.com/docs`. Le certificat SSL est obtenu et renouvele automatiquement.

### 4. Securiser les ports

Par defaut, MLflow (port 5000) et l'API (port 8000) sont exposes publiquement. En production, modifier le `docker-compose.yml` pour les restreindre en local :

```yaml
mlflow:
  ports:
    - "127.0.0.1:5000:5000"  # Accessible uniquement en local

api:
  ports:
    - "127.0.0.1:8000:8000"  # L'API passe par Caddy
```

### 5. Redemarrage automatique

Docker Compose relance automatiquement les conteneurs au reboot si vous ajoutez une politique de restart :

```yaml
services:
  mlflow:
    restart: unless-stopped
    # ...
  api:
    restart: unless-stopped
    # ...
```

### 6. Mettre a jour l'application

```bash
cd ia_cmv
git pull
docker compose up --build -d
curl -X POST http://localhost:8000/reload-model
```

## Lancer les tests

```bash
cd app
python -m pytest test_main.py -v
```

## Dependances

- FastAPI 0.124.2
- Uvicorn 0.38.0
- Pydantic 2.10.6
- scikit-learn 1.6.1
- XGBoost 3.1.2
- numpy 2.3.5
- MLflow 2.21.3
- pytest 9.0.2

## Auteur

Yann Clain
