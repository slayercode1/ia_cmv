# Utiliser une image Python officielle
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copier le code de l'application et le modèle
COPY main.py .
COPY model_multiple2.pkl .

# Exposer le port 8000
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
