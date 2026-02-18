# Utiliser une image Python légère
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier et installer les dépendances en premier
# (Docker met en cache cette étape si requirements.txt ne change pas)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r /app/requirements.txt

# Copier le code de l'application
# Le modèle est chargé depuis MLflow Model Registry (pas de pickle)
COPY ./app/main.py .

# Exposer le port 8000 pour accéder à l'API
EXPOSE 8000

# Lancer le serveur uvicorn au démarrage du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
