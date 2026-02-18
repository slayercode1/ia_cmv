"""
API de Prédiction de Maintenance de Véhicules
----------------------------------------------
Cette API utilise un modèle XGBoost pour prédire
le nombre de jours avant la prochaine maintenance d'un véhicule.

Le modèle est chargé depuis MLflow Model Registry.

Auteur : Yann Clain
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# --- Création de l'application FastAPI ---
app = FastAPI(
    title="API Maintenance Véhicules",
    description="Prédit le nombre de jours avant la prochaine maintenance d'un véhicule.",
    version="2.0.0",
)

# --- Variables globales pour le modèle ---
pipeline = None
rmse = None
model_source = "non chargé"


def load_model_from_mlflow():
    """
    Charge le modèle depuis MLflow Model Registry.
    Nécessite les variables d'environnement MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_ALIAS.
    Retourne (pipeline, rmse, description_source).
    """
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ.get("MODEL_NAME", "maintenance_vehicules")
    model_alias = os.environ.get("MODEL_ALIAS", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Charger le pipeline depuis le registry avec l'alias "champion"
    model_uri = f"models:/{model_name}@{model_alias}"
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)

    # Récupérer le RMSE depuis les tags de la version du modèle
    model_version = client.get_model_version_by_alias(model_name, model_alias)
    rmse_value = float(model_version.tags.get("rmse", 15.26))

    source = f"MLflow ({model_name}@{model_alias}, version {model_version.version})"
    return loaded_pipeline, rmse_value, source


def load_model():
    """
    Charge le modèle depuis MLflow Model Registry.
    Nécessite la variable d'environnement MLFLOW_TRACKING_URI.
    """
    global pipeline, rmse, model_source

    if "MLFLOW_TRACKING_URI" not in os.environ:
        print("MLFLOW_TRACKING_URI non défini. Définissez cette variable pour charger le modèle.")
        pipeline = None
        rmse = None
        model_source = "non chargé (MLFLOW_TRACKING_URI non défini)"
        return

    try:
        pipeline, rmse, model_source = load_model_from_mlflow()
        print(f"Modèle chargé depuis {model_source}")
    except Exception as e:
        print(f"Impossible de charger le modèle depuis MLflow : {e}")
        print("Le modèle sera disponible après enregistrement dans le registry.")
        pipeline = None
        rmse = None
        model_source = "non chargé (MLflow indisponible)"


# --- Charger le modèle au démarrage ---
load_model()


# --- Liste des features dans l'ordre attendu par le modèle ---
# IMPORTANT : cet ordre doit correspondre à celui utilisé lors de l'entrainement
FEATURE_NAMES = [
    "km_actuel",
    "km_moyen_annuel",
    "km_derniere_revision",
    "jours_depuis_derniere_revision",
    "km_depuis_derniere_revision",
    "nb_revisions_effectuees",
    "intervalle_recommande_jours",
    "intervalle_recommande_km",
    "condition_vehicule",
    "nb_pannes_historique",
    "age_vehicule",
    "taux_utilisation_km",
    "taux_utilisation_jours",
    "revisions_par_an",
    "Carburant_Factor",
    "Usage_Factor",
]


# --- Modèle de données d'entrée ---
# Pydantic valide automatiquement les données envoyées par l'utilisateur.
# "Field" permet d'ajouter une description et des exemples pour la documentation.
class InputData(BaseModel):
    km_actuel: float = Field(description="Kilométrage actuel du véhicule", examples=[100000])
    km_moyen_annuel: float = Field(description="Kilomètres parcourus par an en moyenne", examples=[15000])
    km_derniere_revision: float = Field(description="Kilométrage lors de la dernière révision", examples=[95000])
    jours_depuis_derniere_revision: int = Field(description="Nombre de jours depuis la dernière révision", examples=[150])
    km_depuis_derniere_revision: float = Field(description="Kilomètres parcourus depuis la dernière révision", examples=[5000])
    nb_revisions_effectuees: int = Field(description="Nombre total de révisions effectuées", examples=[10])
    intervalle_recommande_jours: int = Field(description="Intervalle recommandé entre révisions (en jours)", examples=[365])
    intervalle_recommande_km: float = Field(description="Intervalle recommandé entre révisions (en km)", examples=[15000])
    condition_vehicule: int = Field(description="Etat du véhicule de 1 (mauvais) à 10 (excellent)", examples=[8])
    nb_pannes_historique: int = Field(description="Nombre de pannes dans l'historique", examples=[1])
    age_vehicule: int = Field(description="Age du véhicule en années", examples=[7])
    taux_utilisation_km: float = Field(description="Taux d'utilisation kilométrique", examples=[1.2])
    taux_utilisation_jours: float = Field(description="Taux d'utilisation en jours", examples=[1.0])
    revisions_par_an: float = Field(description="Nombre moyen de révisions par an", examples=[1.5])
    Carburant_Factor: float = Field(description="Facteur carburant (Essence=1.0, Diesel=1.1, Hybride=1.2, Electrique=1.5)", examples=[1.0])
    Usage_Factor: float = Field(description="Facteur usage (Flotte=0.8, Professionnel=0.9, Personnel=1.2)", examples=[1.2])


def get_recommendation(days: float) -> str:
    """
    Retourne un message de recommandation en fonction du nombre de jours estimé.

    Règles :
    - Moins de 60 jours  -> maintenance urgente
    - 60 à 150 jours     -> maintenance à prévoir bientôt
    - 150 à 270 jours    -> maintenance à planifier
    - Plus de 270 jours  -> pas urgent
    """
    if days < 60:
        return "Planifier la maintenance très prochainement (< 2 mois)"
    elif days < 150:
        return "Planifier la maintenance dans les 2 à 5 mois"
    elif days < 270:
        return "Planifier la maintenance dans 5 à 9 mois"
    else:
        return "La maintenance n'est pas urgente (> 9 mois)"


# --- Route d'accueil ---
# Permet de vérifier que l'API fonctionne (health check)
@app.get("/")
async def root():
    return {
        "message": "API de prédiction de maintenance véhicules",
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_source": model_source,
    }


# --- Route de rechargement du modèle ---
@app.post("/reload-model")
async def reload_model():
    """
    Recharge le modèle depuis MLflow Model Registry.
    Utile après avoir enregistré un nouveau modèle dans le registry via le notebook.
    """
    load_model()
    return {
        "model_loaded": pipeline is not None,
        "model_source": model_source,
    }


# --- Route de prédiction ---
@app.post("/predict")
async def predict(data: InputData):
    """
    Prédit le nombre de jours avant la prochaine maintenance.

    Etapes :
    1. Récupérer les 16 features envoyées par l'utilisateur
    2. Faire la prédiction avec le pipeline (normalisation + modèle)
    3. Calculer une fourchette de confiance avec le RMSE
    4. Générer une recommandation
    """
    # Vérifier que le modèle est chargé
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé. "
                   "Veuillez d'abord enregistrer un modèle dans MLflow "
                   "puis appeler POST /reload-model.",
        )

    # Etape 1 : Transformer les données en tableau numpy
    # On utilise la liste FEATURE_NAMES pour garder le bon ordre
    features = np.array([[getattr(data, name) for name in FEATURE_NAMES]])

    # Etape 2 : Prédiction avec le pipeline
    # Le pipeline fait automatiquement : normalisation (scaler) puis prédiction (modèle)
    days = float(pipeline.predict(features)[0])

    # Etape 3 : Calculer la fourchette de confiance
    # On utilise 2x le RMSE pour avoir un intervalle de ~95%
    lower_bound = max(0, days - 2 * rmse)  # Minimum 0 jours
    upper_bound = days + 2 * rmse

    # Etape 4 : Générer la recommandation
    recommendation = get_recommendation(days)

    # Retourner les résultats en JSON
    return {
        "estimation_jours": round(days, 0),
        "estimation_mois": round(days / 30, 1),
        "fourchette_min_mois": round(lower_bound / 30, 1),
        "fourchette_max_mois": round(upper_bound / 30, 1),
        "recommandation": recommendation,
    }
