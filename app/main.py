from typing import Union
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Chemin du modèle qui fonctionne en local et dans Docker
model_path = Path(__file__).parent / 'model_multiple2.pkl'
if not model_path.exists():
    model_path = Path(__file__).parent.parent / 'notebook' / 'model_multiple2.pkl'

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
rmse_multi = model_data.get('rmse', 15.2565)


class InputData(BaseModel):
    km_actuel: float
    km_moyen_annuel: float
    km_derniere_revision: float
    jours_depuis_derniere_revision: int
    km_depuis_derniere_revision: float
    nb_revisions_effectuees: int
    intervalle_recommande_jours: int
    intervalle_recommande_km: float
    condition_vehicule: int
    nb_pannes_historique: int
    age_vehicule: int
    taux_utilisation_km: float
    taux_utilisation_jours: float
    revisions_par_an: float
    Carburant_Factor: float
    Usage_Factor: float


@app.post("/predict")
async def predict(data: InputData):
    # Préparer les features
    features = [[
        data.km_actuel,
        data.km_moyen_annuel,
        data.km_derniere_revision,
        data.jours_depuis_derniere_revision,
        data.km_depuis_derniere_revision,
        data.nb_revisions_effectuees,
        data.intervalle_recommande_jours,
        data.intervalle_recommande_km,
        data.condition_vehicule,
        data.nb_pannes_historique,
        data.age_vehicule,
        data.taux_utilisation_km,
        data.taux_utilisation_jours,
        data.revisions_par_an,
        data.Carburant_Factor,
        data.Usage_Factor
    ]]

    # Normaliser et prédire
    features_scaled = scaler.transform(features)
    days = float(model.predict(features_scaled)[0])

    # Calcul de l'intervalle
    lower_bound = max(0, days - 2 * rmse_multi)
    upper_bound = days + 2 * rmse_multi

    # Logique de recommandation simple basée sur l'estimation
    if days < 60:
        recommendation = "Planifier la maintenance très prochainement (< 2 mois)"
    elif days < 150:
        recommendation = "Planifier la maintenance dans les 2 à 5 mois"
    elif days < 270:
        recommendation = "Planifier la maintenance dans 5 à 9 mois"
    else:
        recommendation = "La maintenance n'est pas urgente (> 9 mois)"

    # Retourner les résultats
    return {
        "estimation_jours": round(days, 0),
        "estimation_mois": round(days / 30, 1),
        "fourchette_min_mois": round(lower_bound / 30, 1),
        "fourchette_max_mois": round(upper_bound / 30, 1),
        "recommandation": recommendation
    }
