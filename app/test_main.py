"""
Tests de l'API de Prédiction de Maintenance
--------------------------------------------
Ce fichier contient tous les tests pour vérifier que l'API fonctionne correctement.

Le modèle MLflow est mocké pour que les tests s'exécutent sans serveur MLflow.

Pour lancer les tests : pytest app/test_main.py -v
"""

from unittest.mock import MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient

# Mocker le modèle avant d'importer l'application
# On crée un faux pipeline qui retourne toujours 150 jours
import app.main as main

mock_pipeline = MagicMock()
mock_pipeline.predict.return_value = np.array([150.0])
main.pipeline = mock_pipeline
main.rmse = 15.0
main.model_source = "mock (tests)"

client = TestClient(main.app)

# --- Données de test réutilisables ---
# On crée un payload "de base" qu'on peut modifier dans chaque test.
# Cela évite de copier-coller les mêmes 16 champs partout.
BASE_PAYLOAD = {
    "km_actuel": 50000,
    "km_moyen_annuel": 15000,
    "km_derniere_revision": 45000,
    "jours_depuis_derniere_revision": 120,
    "km_depuis_derniere_revision": 5000,
    "nb_revisions_effectuees": 5,
    "intervalle_recommande_jours": 365,
    "intervalle_recommande_km": 15000,
    "condition_vehicule": 3,
    "nb_pannes_historique": 2,
    "age_vehicule": 3,
    "taux_utilisation_km": 0.33,
    "taux_utilisation_jours": 0.33,
    "revisions_par_an": 1.67,
    "Carburant_Factor": 1.0,
    "Usage_Factor": 2.0,
}

# Les 4 recommandations possibles retournées par l'API
RECOMMANDATIONS = [
    "Planifier la maintenance très prochainement (< 2 mois)",
    "Planifier la maintenance dans les 2 à 5 mois",
    "Planifier la maintenance dans 5 à 9 mois",
    "La maintenance n'est pas urgente (> 9 mois)",
]


def make_payload(**overrides):
    """
    Crée un payload de test en partant de BASE_PAYLOAD.
    On peut remplacer n'importe quel champ avec des arguments nommés.

    Exemple : make_payload(km_actuel=200000, condition_vehicule=2)
    """
    payload = BASE_PAYLOAD.copy()
    payload.update(overrides)
    return payload


# ========================================
# Tests de la route d'accueil
# ========================================

def test_root():
    """Vérifie que la route d'accueil (/) fonctionne"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ========================================
# Tests de la route /predict
# ========================================

def test_predict_valid_input():
    """Test avec des données d'entrée valides"""
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.status_code == 200

    data = response.json()

    # Vérifier que tous les champs sont présents dans la réponse
    assert "estimation_jours" in data
    assert "estimation_mois" in data
    assert "fourchette_min_mois" in data
    assert "fourchette_max_mois" in data
    assert "recommandation" in data

    # Vérifier les types de données
    assert isinstance(data["estimation_jours"], (int, float))
    assert isinstance(data["estimation_mois"], (int, float))
    assert isinstance(data["fourchette_min_mois"], (int, float))
    assert isinstance(data["fourchette_max_mois"], (int, float))
    assert isinstance(data["recommandation"], str)

    # Vérifier la cohérence : min <= estimation <= max
    assert data["estimation_jours"] >= 0
    assert data["fourchette_min_mois"] <= data["estimation_mois"] <= data["fourchette_max_mois"]


def test_predict_recommendation_is_valid():
    """Vérifie que la recommandation fait partie des catégories connues"""
    response = client.post("/predict", json=BASE_PAYLOAD)
    data = response.json()
    assert data["recommandation"] in RECOMMANDATIONS


def test_predict_recommendation_categories():
    """Vérifie les différentes catégories de recommandation"""
    # < 60 jours
    mock_pipeline.predict.return_value = np.array([30.0])
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.json()["recommandation"] == RECOMMANDATIONS[0]

    # 60 à 150 jours
    mock_pipeline.predict.return_value = np.array([100.0])
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.json()["recommandation"] == RECOMMANDATIONS[1]

    # 150 à 270 jours
    mock_pipeline.predict.return_value = np.array([200.0])
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.json()["recommandation"] == RECOMMANDATIONS[2]

    # > 270 jours
    mock_pipeline.predict.return_value = np.array([300.0])
    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.json()["recommandation"] == RECOMMANDATIONS[3]

    # Remettre la valeur par défaut
    mock_pipeline.predict.return_value = np.array([150.0])


# ========================================
# Tests de validation des données
# ========================================

def test_predict_missing_field():
    """Test avec un champ manquant -> doit retourner une erreur 422"""
    payload = make_payload()
    del payload["km_derniere_revision"]  # On supprime un champ obligatoire

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Erreur de validation Pydantic


def test_predict_invalid_type():
    """Test avec un type de données invalide -> doit retourner une erreur 422"""
    payload = make_payload(km_actuel="invalide")  # String au lieu de float

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ========================================
# Tests avec des valeurs extrêmes
# ========================================

def test_predict_negative_values():
    """Test avec des valeurs négatives (cas limites)"""
    payload = make_payload(km_actuel=-1000)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    # La fourchette minimale ne doit jamais être négative
    assert data["fourchette_min_mois"] >= 0


def test_predict_zero_values():
    """Test avec toutes les valeurs à zéro"""
    payload = {key: 0 if isinstance(val, int) else 0.0 for key, val in BASE_PAYLOAD.items()}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "estimation_jours" in data


def test_predict_high_values():
    """Test avec des valeurs très élevées"""
    payload = make_payload(
        km_actuel=500000,
        km_moyen_annuel=50000,
        km_derniere_revision=495000,
        jours_depuis_derniere_revision=400,
        nb_revisions_effectuees=50,
        condition_vehicule=5,
        nb_pannes_historique=10,
        age_vehicule=10,
        taux_utilisation_km=5.0,
        taux_utilisation_jours=1.1,
        revisions_par_an=5.0,
        Carburant_Factor=3.0,
        Usage_Factor=3.0,
    )

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "estimation_jours" in data


# ========================================
# Test du modèle non chargé
# ========================================

def test_predict_model_not_loaded():
    """Test que l'API retourne 503 si le modèle n'est pas chargé"""
    original_pipeline = main.pipeline
    main.pipeline = None

    response = client.post("/predict", json=BASE_PAYLOAD)
    assert response.status_code == 503

    main.pipeline = original_pipeline
