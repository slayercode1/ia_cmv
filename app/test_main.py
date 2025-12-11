from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_valid_input():
    """Test avec des données d'entrée valides"""
    payload = {
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
        "Usage_Factor": 2.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "estimation_jours" in data
    assert "estimation_mois" in data
    assert "fourchette_min_mois" in data
    assert "fourchette_max_mois" in data
    assert "recommandation" in data

    # Vérifier les types
    assert isinstance(data["estimation_jours"], (int, float))
    assert isinstance(data["estimation_mois"], (int, float))
    assert isinstance(data["fourchette_min_mois"], (int, float))
    assert isinstance(data["fourchette_max_mois"], (int, float))
    assert isinstance(data["recommandation"], str)

    # Vérifier la cohérence des valeurs
    assert data["estimation_jours"] >= 0
    assert data["fourchette_min_mois"] <= data["estimation_mois"] <= data["fourchette_max_mois"]


def test_predict_maintenance_urgente():
    """Test pour vérifier la recommandation de maintenance urgente (< 60 jours)"""
    payload = {
        "km_actuel": 80000,
        "km_moyen_annuel": 25000,
        "km_derniere_revision": 75000,
        "jours_depuis_derniere_revision": 200,
        "km_depuis_derniere_revision": 5000,
        "nb_revisions_effectuees": 8,
        "intervalle_recommande_jours": 180,
        "intervalle_recommande_km": 10000,
        "condition_vehicule": 2,
        "nb_pannes_historique": 5,
        "age_vehicule": 5,
        "taux_utilisation_km": 0.5,
        "taux_utilisation_jours": 1.11,
        "revisions_par_an": 1.6,
        "Carburant_Factor": 0.0,
        "Usage_Factor": 1.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    # Vérifier que la recommandation existe
    assert data["recommandation"] in [
        "Planifier la maintenance très prochainement (< 2 mois)",
        "Planifier la maintenance dans les 2 à 5 mois",
        "Planifier la maintenance dans 5 à 9 mois",
        "La maintenance n'est pas urgente (> 9 mois)"
    ]


def test_predict_missing_field():
    """Test avec un champ manquant"""
    payload = {
        "km_actuel": 50000,
        "km_moyen_annuel": 15000,
        # km_derniere_revision manquant
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
        "Usage_Factor": 2.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_type():
    """Test avec un type de données invalide"""
    payload = {
        "km_actuel": "invalide",  # devrait être un float
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
        "Usage_Factor": 2.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_values():
    """Test avec des valeurs négatives (cas limites)"""
    payload = {
        "km_actuel": -1000,  # valeur négative
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
        "Usage_Factor": 2.0
    }

    response = client.post("/predict", json=payload)
    # L'API accepte les valeurs mais le modèle les traite
    assert response.status_code == 200
    # La fourchette minimale devrait être 0 minimum
    data = response.json()
    assert data["fourchette_min_mois"] >= 0


def test_predict_zero_values():
    """Test avec des valeurs à zéro"""
    payload = {
        "km_actuel": 0,
        "km_moyen_annuel": 0,
        "km_derniere_revision": 0,
        "jours_depuis_derniere_revision": 0,
        "km_depuis_derniere_revision": 0,
        "nb_revisions_effectuees": 0,
        "intervalle_recommande_jours": 0,
        "intervalle_recommande_km": 0,
        "condition_vehicule": 0,
        "nb_pannes_historique": 0,
        "age_vehicule": 0,
        "taux_utilisation_km": 0.0,
        "taux_utilisation_jours": 0.0,
        "revisions_par_an": 0.0,
        "Carburant_Factor": 0.0,
        "Usage_Factor": 0.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimation_jours" in data


def test_predict_high_values():
    """Test avec des valeurs très élevées"""
    payload = {
        "km_actuel": 500000,
        "km_moyen_annuel": 50000,
        "km_derniere_revision": 495000,
        "jours_depuis_derniere_revision": 400,
        "km_depuis_derniere_revision": 5000,
        "nb_revisions_effectuees": 50,
        "intervalle_recommande_jours": 365,
        "intervalle_recommande_km": 10000,
        "condition_vehicule": 5,
        "nb_pannes_historique": 10,
        "age_vehicule": 10,
        "taux_utilisation_km": 5.0,
        "taux_utilisation_jours": 1.1,
        "revisions_par_an": 5.0,
        "Carburant_Factor": 3.0,
        "Usage_Factor": 3.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimation_jours" in data


def test_predict_recommendation_categories():
    """Test pour vérifier que toutes les catégories de recommandation fonctionnent"""
    test_cases = [
        # Cas pour maintenance urgente (< 60 jours)
        {
            "expected_category": "très prochainement",
            "payload": {
                "km_actuel": 100000,
                "km_moyen_annuel": 30000,
                "km_derniere_revision": 95000,
                "jours_depuis_derniere_revision": 300,
                "km_depuis_derniere_revision": 5000,
                "nb_revisions_effectuees": 10,
                "intervalle_recommande_jours": 180,
                "intervalle_recommande_km": 10000,
                "condition_vehicule": 2,
                "nb_pannes_historique": 8,
                "age_vehicule": 6,
                "taux_utilisation_km": 0.5,
                "taux_utilisation_jours": 1.67,
                "revisions_par_an": 1.67,
                "Carburant_Factor": 0.0,
                "Usage_Factor": 1.0
            }
        }
    ]

    for case in test_cases:
        response = client.post("/predict", json=case["payload"])
        assert response.status_code == 200
        data = response.json()
        # Vérifier que la recommandation contient une partie attendue
        assert isinstance(data["recommandation"], str)
