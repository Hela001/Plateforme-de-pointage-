from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import cv2
import face_recognition
import pyttsx3
import numpy as np
import io
from PIL import Image
from pymongo import MongoClient
from bson.binary import Binary
from bson import ObjectId 
import logging
from datetime import datetime, time, timedelta
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor
import base64
import os
import uuid
from bson.regex import Regex
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF pour PDF
from reports_utils import embedding_model, generate_daily_reports, index_daily_reports, search_reports
from openai import OpenAI
from dotenv import load_dotenv
import httpx
import re
import json
from reports_utils import index_additional_docs
from fraude_detector import analyser_video  # la fonction principale de fraude_detector.py
from reports_utils import (extract_text_from_pdf,extract_text_from_excel,extract_text_from_docx,extract_text_from_image,init_faiss_index,load_meta,index_report,)
# Indexation complète au démarrage
from reports_utils import index_all_reports
index_all_reports()
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from datetime import timezone

SYSTEM_PROMPT_CLASSIFICATION_TEMPLATE = """
Tu es un assistant intelligent chargé de classifier l’intention d’une requête utilisateur.

Classes possibles d’intention (uniquement ces choix) :
- salutation
- question_bien_etre
- question_meteo
- question_generale
- demande_rapport
- question_metier
- autre

Instructions :
1) Classifie la requête dans l’une des classes ci-dessus.
2) Fournis une réponse adaptée selon la classe.
3) Respecte strictement ce format JSON (sans autre texte) :

{{
  "intention": "nom_de_l_intention",
  "response": "texte de la réponse adaptée"
}}

Répond uniquement en français.

Exemples :
Utilisateur : "Salut, ça va ?"
Réponse JSON : {{
  "intention": "salutation",
  "response": "Bonjour ! Comment puis-je vous aider ?"
}}

Utilisateur : "Quelle est la météo aujourd’hui ?"
Réponse JSON : {{
  "intention": "question_meteo",
  "response": "Je ne dispose pas des données météo actuelles, mais je peux vous aider sur d'autres sujets."
}}

Utilisateur : "Peux-tu me faire un résumé des fraudes ?"
Réponse JSON : {{
  "intention": "demande_rapport",
  "response": "Je cherche les rapports correspondants..."
}}

Maintenant, analyse la requête utilisateur suivante et répond en JSON :

\"\"\" 
{user_query} 
\"\"\"
"""

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "groq/compound"
FACE_RECOG_THRESHOLD = 0.65  # distance max pour accepter une correspondance
load_dotenv()

# --- Configuration globale ---
FILES_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(FILES_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app, supports_credentials=True)
executor = ThreadPoolExecutor(max_workers=5)  # 5 threads simultanés pour l'envoi des mails
# Indexation initiale
index_additional_docs("knowledge_base")

# Réindexation toutes les 10 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: index_additional_docs("knowledge_base"), 'interval', minutes=10)
scheduler.start()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Clé API non définie")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)
def mongo_connection():
    try:
        client = MongoClient("mongodb+srv://stage:stage@cluster0.gbm1c.mongodb.net/stage?retryWrites=true&w=majority")
        return client["stage"]
    except Exception as e:
        logging.error(f"Erreur de connexion à MongoDB: {e}")
        return None
db = mongo_connection()
if db is None:
    raise ValueError("Impossible de se connecter à MongoDB")

def detect_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)
    return locations, encodings

def recognize_person(encoding, known_faces, known_names):
    matches = face_recognition.compare_faces(known_faces, encoding)
    if True in matches:
        index = matches.index(True)
        return {"username": known_names[index]}
    return None

def reverse_geocode(lat, lon):
    try:
        url = 'https://nominatim.openstreetmap.org/reverse'
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 18, 'addressdetails': 1}
        headers = {'User-Agent': 'mon-app-pointage/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('display_name', 'Adresse inconnue')
        else:
            logging.warning(f"Geocoding error HTTP {response.status_code}")
            return 'Erreur géocodage'
    except Exception as e:
        logging.error(f"Erreur géocodage inverse: {e}")
        return 'Erreur géocodage'

def load_known_faces_by_role(role):
    db = mongo_connection()
    if db is None:
        return [], [], {}

    known_faces, known_names, known_user_ids = [], [], {}

    try:
        users = db.user.find({"photo": {"$exists": True}, "role": role})
        for user in users:
            image_data = user.get('photo')
            if not image_data:
                continue
            image_bytes = bytes(image_data) if isinstance(image_data, Binary) else image_data
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_np = np.array(img)
            encodings = face_recognition.face_encodings(img_np)
            if not encodings:
                continue
            encoding = encodings[0]
            username = user.get("username", "Inconnu")
            known_faces.append(encoding)
            known_names.append(username)
            known_user_ids[username] = str(user.get("_id"))
    except Exception as e:
        logging.error(f"Erreur MongoDB : {e}")

    return known_faces, known_names, known_user_ids

def capture_video(duration=3):
    output_path = f"fraude_check_{uuid.uuid4().hex}.avi"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Impossible d'ouvrir la caméra.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def analyser_fraude(video_path, username=None):
    """
    Analyse la vidéo pour détecter une fraude.

    Retourne toujours un tuple : (fraude_detectee: bool, details: dict)
    """
    try:
        details = analyser_video(video_path)  # dict depuis fraude_detector.py
        fraude_detectee = details.get("fraude", False)
        logging.info(f"Résultat analyse fraude: {details}")
        return fraude_detectee, details
    except Exception as e:
        logging.error(f"Erreur analyse fraude: {e}")
        return False, {"fraude": False, "reasons": ["Erreur analyse locale"]}

def verifier_depart_non_autorise_pour_jour_precedent(db, user_id, username, role, adresse):
    hier = datetime.now() - timedelta(days=1)
    jour_prec_debut = datetime(hier.year, hier.month, hier.day)
    jour_prec_fin = jour_prec_debut + timedelta(days=1)

    pointage_collection = db.pointage
    notification_collection = db.notification

    dernier_depart = pointage_collection.find_one(
        {
            "user_id": user_id,
            "date_pointage": {"$gte": jour_prec_debut, "$lt": jour_prec_fin},
            "statut": "depart"
        },
        sort=[("heure_depart", -1)]
    )
    if not dernier_depart:
        return

    heure_depart = dernier_depart.get("heure_depart") or dernier_depart.get("heure_pointage")
    if not heure_depart:
        return

    heure_depart_time = heure_depart.time()
    limite_jour = time(16, 0)
    limite_nuit = time(4, 0)

    if heure_depart_time < limite_jour or heure_depart_time < limite_nuit:
        exists = notification_collection.find_one({
            "user_id": user_id,
            "date": jour_prec_debut,
            "statut": "depart_non_autorise"
        })
        if exists:
            return

        notification_doc = {
            "user_id": user_id,
            "username": username,
            "date": jour_prec_debut,
            "heure": heure_depart,
            "statut": "depart_non_autorise",
            "message": f"Dernier départ hier non autorisé à {heure_depart.strftime('%H:%M:%S')}",
            "role": role,
            "adresse": adresse
        }
        notification_collection.insert_one(notification_doc)
        logging.info(f"Notification départ non autorisé créée pour user {username} hier.")

@app.route('/get_person_data', methods=['POST'])
def get_person_data():
    video_capture = None
    engine = None
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        if None in [lat, lon]:
            return jsonify({"message": "Coordonnées manquantes"}), 400
        db = mongo_connection()
        if db is None:
            return jsonify({"message": "Erreur base de données"}), 500
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return jsonify({"message": "Erreur caméra"}), 500
        engine = pyttsx3.init()
        ret, frame = video_capture.read()
        if not ret:
            return jsonify({"message": "Erreur capture image"}), 500
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({"message": "Erreur encodage image"}), 500
        image_bytes = jpeg.tobytes()
        locations, encodings = detect_faces(frame)
        if not encodings:
            engine.say("Aucun visage détecté")
            engine.runAndWait()
            return jsonify({"statut": "non_reconnu", "message": "Aucun visage détecté"}), 200
        for encoding in encodings:
            for role in ["SITE_SUPERVISOR", "EMPLOYEE", "ADMIN"]:
                known_faces, known_names, known_user_ids = load_known_faces_by_role(role)
                user = recognize_person(encoding, known_faces, known_names)
                if user:
                    user_id_str = known_user_ids[user["username"]]
                    adresse = reverse_geocode(lat, lon)
                    video_path = capture_video()
                    fraude_detectee, details = analyser_fraude(video_path, user["username"])
                    if fraude_detectee:
                        db.fraude.insert_one({
                            "user_id": user_id_str,
                            "username": user["username"],
                            "date": now,
                            "role": role,
                            "adresse": adresse,
                            "message": "Fraude détectée lors du pointage",
                            "raisons": details.get("raisons", [])
                        })
                        db.notification.insert_one({
                            "user_id": user_id_str,
                            "username": user["username"],
                            "date": now,
                            "heure": now,
                            "statut": "fraude",
                            "role": role,
                            "adresse": adresse,
                            "message": "Une fraude a été détectée (vidéo suspecte)"
                        })
                        rh_users = db.user.find({"role": "RH", "email": {"$exists": True}})
                        for rh in rh_users:
                            email = rh.get("email")
                            envoyer_mail_fraude_async(
                                destinataire=email,
                                nom_utilisateur=user["username"],
                                type_fraude="Pointage suspect",
                                date_fraude=now.strftime("%Y-%m-%d %H:%M:%S"),
                                raisons=details.get("raisons", [])
                            )
                        logging.warning(f"🚨 Fraude détectée pour {user['username']}")
                    else:
                        logging.info(f"Pas de fraude détectée pour {user['username']}")
                    try:
                        if fraude_detectee:
                            inspection_dir = "fraudes_inspection"
                            os.makedirs(inspection_dir, exist_ok=True)
                            os.rename(video_path, os.path.join(inspection_dir, os.path.basename(video_path)))
                        else:
                            os.remove(video_path)
                    except Exception as e:
                        logging.warning(f"Erreur suppression fichier vidéo temporaire: {e}")
                    pointage_collection = db.pointage
                    pointage_doc = pointage_collection.find_one({
                        "user_id": user_id_str,
                        "date_pointage": {"$gte": today_start, "$lt": today_start + timedelta(days=1)}
                    })
                    if not pointage_doc:
                        statut = "arrivee"
                    else:
                        nb_arrivees = len(pointage_doc.get("arrivees", []))
                        nb_departs = len(pointage_doc.get("departs", []))
                        statut = "arrivee" if nb_arrivees <= nb_departs else "depart"
                    if statut == "arrivee":
                        verifier_depart_non_autorise_pour_jour_precedent(
                            db, user_id_str, user["username"], role, adresse
                        )
                        pointage_collection.update_one(
                            {
                                "user_id": user_id_str,
                                "date_pointage": {"$gte": today_start, "$lt": today_start + timedelta(days=1)}
                            },
                            {
                                "$setOnInsert": {
                                    "user_id": user_id_str,
                                    "username": user["username"],
                                    "date_pointage": today_start,
                                    "role": role
                                },
                                "$push": {"arrivees": {"heure": now}},
                                "$set": {
                                    "image": Binary(image_bytes),
                                    "adresse": adresse,
                                    "localisation": {"lat": lat, "lon": lon}
                                }
                            },
                            upsert=True
                        )
                    else:  # depart
                        pointage_collection.update_one(
                            {
                                "user_id": user_id_str,
                                "date_pointage": {"$gte": today_start, "$lt": today_start + timedelta(days=1)}
                            },
                            {
                                "$push": {"departs": {"heure": now}},
                                "$set": {
                                    "adresse": adresse,
                                    "localisation": {"lat": lat, "lon": lon}
                                }
                            }
                        )

                    final_doc = pointage_collection.find_one({
                        "user_id": user_id_str,
                        "date_pointage": {"$gte": today_start, "$lt": today_start + timedelta(days=1)}
                    })
                    heures_arrivee = [a["heure"].strftime("%H:%M:%S") for a in final_doc.get("arrivees", [])]
                    heures_depart = [d["heure"].strftime("%H:%M:%S") for d in final_doc.get("departs", [])]

                    engine.say(f"{user['username']}, statut {statut} enregistré.")
                    engine.runAndWait()
                    return jsonify({
                        "username": user["username"],
                        "statut": statut,
                        "adresse": adresse,
                        "role": role,
                        "heures_arrivee": heures_arrivee,
                        "heures_depart": heures_depart,
                        "fraude": fraude_detectee,
                        "raisons": ["Fraude vidéo détectée"] if fraude_detectee else []
                    }), 200
        engine.say("Visage non reconnu.")
        engine.runAndWait()
        return jsonify({"statut": "non_reconnu", "message": "Visage non reconnu."}), 200
    except Exception as e:
        logging.error(f"Erreur reconnaissance : {e}")
        return jsonify({"message": "Erreur serveur"}), 500
    finally:
        if video_capture:
            video_capture.release()
        if engine:
            engine.stop()
@app.route('/fraudes', methods=['GET'])
def get_fraudes():
    db = mongo_connection()
    if db is None:
        return jsonify({"message": "Erreur base de données"}), 500
    try:
        # On récupère les 50 dernières fraudes, triées par date décroissante
        fraudes_cursor = db.fraude.find().sort("date", -1).limit(50)
        fraudes_list = []
        for f in fraudes_cursor:
            fraudes_list.append({
                "id": str(f.get("_id")),
                "user_id": f.get("user_id"),
                "username": f.get("username"),
                "date": f.get("date").isoformat() if f.get("date") else None,
                "role": f.get("role"),
                "adresse": f.get("adresse"),
                "message": f.get("message"),
                "raisons": f.get("raisons", [])
            })
        return jsonify(fraudes_list), 200
    except Exception as e:
        logging.error(f"Erreur récupération fraudes : {e}")
        return jsonify({"message": "Erreur serveur"}), 500
def envoyer_mail_fraude_async(destinataire, nom_utilisateur, type_fraude, date_fraude, raisons):
    executor.submit(envoyer_mail_fraude, destinataire, nom_utilisateur, type_fraude, date_fraude, raisons)
def envoyer_mail_fraude(destinataire, nom_utilisateur, type_fraude, date_fraude, raisons):
    expediteur = "neflahela@gmail.com"
    mot_de_passe = "motdepasseoumotdepasseapplication"
    raisons_text = ", ".join(raisons) if raisons else "Non spécifié"
    sujet = f"⚠️ Fraude détectée - {nom_utilisateur}"
    corps = f"""Bonjour,
Une fraude a été détectée.

👤 Utilisateur : {nom_utilisateur}
📅 Date : {date_fraude}
🚨 Type : {type_fraude}
📌 Raisons : {raisons_text}

Merci de prendre les mesures nécessaires.

Cordialement,
Le système de détection
"""
    message = MIMEMultipart()
    message["From"] = expediteur
    message["To"] = destinataire
    message["Subject"] = sujet
    message.attach(MIMEText(corps, "plain", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as serveur:
            serveur.ehlo()
            serveur.starttls()
            serveur.login(expediteur, mot_de_passe)
            serveur.send_message(message)
        print(f"✅ Mail envoyé avec succès à {destinataire}")
    except smtplib.SMTPAuthenticationError:
        print("❌ Erreur d'authentification SMTP. Vérifie ton mot de passe ou le mot de passe d'application.")
    except smtplib.SMTPException as e:
        print(f"❌ Erreur d'envoi du mail : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
def verifier_depart_non_autorise_interne(check_date=None):
    db = mongo_connection()
    if db is None:
        logging.error("Erreur base de données")
        return

    if check_date is None:
        check_date = datetime.now()
    jour_prec = check_date - timedelta(days=1)
    jour_prec_debut = datetime(jour_prec.year, jour_prec.month, jour_prec.day)
    jour_prec_fin = jour_prec_debut + timedelta(days=1)
    pointage_collection = db.pointage
    notification_collection = db.notification
    results = pointage_collection.find({
        "date_pointage": {"$gte": jour_prec_debut, "$lt": jour_prec_fin},
        "statut": "depart"
    })
    for depart in results:
        heure_depart = depart.get("heure_depart") or depart.get("heure_pointage")
        if not heure_depart:
            continue

        heure_depart_time = heure_depart.time()
        limite = time(16, 0) if time(7, 30) <= heure_depart_time <= time(17, 30) else time(4, 0)

        if heure_depart_time < limite:
            exists = notification_collection.find_one({
                "user_id": depart["user_id"],
                "date": jour_prec_debut,
                "statut": "depart_non_autorise"
            })
            if exists:
                continue

            notification_doc = {
                "user_id": depart["user_id"],
                "username": depart.get("username", ""),
                "date": jour_prec_debut,
                "heure": heure_depart,
                "statut": "depart_non_autorise",
                "message": f"Départ non autorisé à {heure_depart.strftime('%H:%M:%S')}",
                "role": depart.get("role", ""),
                "adresse": depart.get("adresse", "")
            }
            notification_collection.insert_one(notification_doc)
            logging.info(f"Notification départ non autorisé créée pour {depart.get('username', '')}.")

    logging.info("✅ Vérification départs non autorisés terminée")

def verifier_absences_fin_journee():
    db = mongo_connection()
    if db is None:
        logging.error("Erreur base de données")
        return

    user_collection = db.user
    pointage_collection = db.pointage
    notification_collection = db.notification

    now = datetime.now()
    jour_prec = now - timedelta(days=1)
    jour_prec_debut = datetime(jour_prec.year, jour_prec.month, jour_prec.day)
    jour_prec_fin = jour_prec_debut + timedelta(days=1)

    users = list(user_collection.find({"role": {"$ne": "ADMIN"}}))

    for user in users:
        user_id = str(user["_id"])

        count = pointage_collection.count_documents({
            "user_id": user_id,
            "date_pointage": {"$gte": jour_prec_debut, "$lt": jour_prec_fin}
        })

        if count == 0:
            exists = notification_collection.find_one({
                "user_id": user_id,
                "date": jour_prec_debut,
                "statut": "absent"
            })
            if exists:
                continue

            notif = {
                "user_id": user_id,
                "username": user.get("username", "Inconnu"),
                "date": jour_prec_debut,
                "statut": "absent",
                "message": "Absence détectée (pas de pointage sur 24h)",
                "role": user.get("role", ""),
                "adresse": ""
            }
            notification_collection.insert_one(notif)
            logging.info(f"Notification absence créée pour user {user.get('username', 'Inconnu')}")

@app.route('/verifier_depart_non_autorise', methods=['POST'])
def verifier_depart_non_autorise_route():
    try:
        data = request.get_json() or {}
        date_str = data.get("date")
        check_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
        verifier_depart_non_autorise_interne(check_date)
        return jsonify({"message": f"Vérification terminée pour {check_date.strftime('%Y-%m-%d')}"}), 200
    except Exception as e:
        logging.error(f"Erreur route vérification départs : {e}")
        return jsonify({"message": "Erreur serveur"}), 500

@app.route('/verifier_fin_journee', methods=['POST'])
def verifier_fin_journee_route():
    try:
        verifier_absences_fin_journee()
        return jsonify({"message": "Vérification des absences terminée"}), 200
    except Exception as e:
        logging.error(f"Erreur vérification fin journée : {e}")
        return jsonify({"message": "Erreur serveur"}), 500

def verifier_toutes_les_verifications():
    logging.info("Début des vérifications combinées")
    verifier_depart_non_autorise_interne(datetime.now() - timedelta(days=1))
    verifier_absences_fin_journee()
    generate_daily_reports()
    index_daily_reports()
    logging.info("Fin des vérifications combinées")

scheduler = BackgroundScheduler(timezone='Africa/Tunis')
scheduler.add_job(
    func=verifier_toutes_les_verifications,
    trigger='cron',
    hour=6,
    minute=0,
    id='verification_combinee_job',
    name='Vérification départs non autorisés + absences',
    replace_existing=True
)
scheduler.start()
logging.info("🗓️ Planificateur initialisé à 6h00")



# --- Classification intention Groq ---
@app.route("/classify_intent", methods=["POST"])
def classify_intent():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query manquante"}), 400

    prompt = SYSTEM_PROMPT_CLASSIFICATION_TEMPLATE.format(user_query=query)
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": query}],
        "temperature": 0.3,
        "max_tokens": 200,
        "response_format": {"type": "json_object"}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        choices = response.json().get("choices")
        if not choices:
            return jsonify({"error": "Réponse Groq vide ou mal formée"}), 500

        content = choices[0].get("message", {}).get("content", "")
        try:
            classification_result = json.loads(content)
        except json.JSONDecodeError:
            classification_result = {"intention": "autre"}
        return jsonify(classification_result)

    except httpx.HTTPStatusError as e:
        logging.error(f"Erreur HTTP Groq: {e.response.status_code} - {e.response.text}")
        return jsonify({"error": "Erreur HTTP Groq"}), 500
    except httpx.RequestError as e:
        logging.error(f"Erreur requête Groq: {e}")
        return jsonify({"error": "Erreur connexion Groq"}), 500
    except Exception as e:
        logging.error(f"Erreur inattendue Groq: {e}")
        return jsonify({"error": "Erreur inattendue Groq"}), 500


@app.route("/reports/<path:filename>", methods=["GET"])
def download_file_http(filename):
    safe_filename = os.path.basename(filename)
    matched_file = find_exact_report_file(safe_filename)
    if not matched_file:
        return jsonify({"error": "Fichier introuvable"}), 404
    return send_from_directory(FILES_DIR, matched_file, as_attachment=True)

@app.route('/generate_reports', methods=['POST'])
def generate_reports_api():
    generate_daily_reports()
    return jsonify({"message": "Rapports générés avec succès"}), 200

@app.route('/index_reports', methods=['POST'])
def index_reports_api():
    index_daily_reports()
    return jsonify({"message": "Rapports indexés avec succès"}), 200

@app.route('/add_report', methods=['POST'])
def add_report():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['file']
        filename = file.filename
        if filename == "":
            return jsonify({"error": "Nom de fichier vide"}), 400

        os.makedirs(FILES_DIR, exist_ok=True)
        file_path = os.path.join(FILES_DIR, filename)
        file.save(file_path)
        logging.info(f"Fichier sauvegardé : {filename}")

        index = init_faiss_index()
        meta = load_meta()
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext in (".xls", ".xlsx"):
            text = extract_text_from_excel(file_path)
        elif ext == ".docx":
            text = extract_text_from_docx(file.stream)
        elif ext in (".png", ".jpg", ".jpeg"):
            text = extract_text_from_image(file_path)
        else:
            return jsonify({"error": f"Type de fichier non supporté: {ext}"}), 400

        report_id = os.path.splitext(filename)[0]
        index_report(report_id, text, index, meta)
        logging.info(f"Fichier indexé : {filename}")

        return jsonify({"message": f"Fichier {filename} ajouté et indexé avec succès"}), 200

    except Exception as e:
        logging.error(f"Erreur ajout fichier : {e}")
        return jsonify({"error": "Erreur serveur", "detail": str(e)}), 500

@app.route("/download_file", methods=["POST"])
def download_file():
    data = request.json or {}
    file_name = data.get("file_name")
    if not file_name:
        return jsonify({"error": "Nom de fichier requis"}), 400

    safe_name = os.path.basename(file_name)
    file_path = os.path.join(FILES_DIR, safe_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "Fichier introuvable"}), 404

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({"fileName": safe_name, "fileData": encoded})

# --- Fonctions utilitaires ---
def find_exact_report_file(filename: str):
    if not os.path.exists(FILES_DIR):
        return None
    for fname in os.listdir(FILES_DIR):
        if fname.lower() == filename.lower():
            return fname
    return None

def get_download_link(filename: str):
    matched_file = find_exact_report_file(filename)
    if matched_file:
        return url_for("download_file_http", filename=matched_file, _external=True)
    return None

def find_report_files(query: str):
    if not os.path.exists(FILES_DIR):
        return []
    norm_query = re.sub(r'[^a-z0-9]', '', query.lower())  # enlève tout sauf lettres/chiffres
    matched_files = []
    for fname in os.listdir(FILES_DIR):
        norm_fname = re.sub(r'[^a-z0-9]', '', fname.lower())
        if any(q_part in norm_fname for q_part in norm_query.split()):
            matched_files.append({
                "filename": fname,
                "snippet": "",
                "file_link": f"/reports/{fname}",
                "download_link": get_download_link(fname),
                "score": 1.0
            })
    return matched_files

# --- Chat & reconnaissance ---
def identify_person_from_file(file_path, threshold=0.65):
    try:
        img = Image.open(file_path).convert('RGB')
        img_np = np.array(img)
        encodings = face_recognition.face_encodings(img_np)
        if not encodings:
            return {"username": "Inconnu", "name": "Inconnu",
                    "department": "Inconnu", "hire_date": "Inconnu", "salary": "Inconnu"}
        unknown_encoding = encodings[0]
        for role in ["SITE_SUPERVISOR", "EMPLOYEE", "ADMIN"]:
            known_faces, known_names, known_user_ids = load_known_faces_by_role(role)
            for i, known_face in enumerate(known_faces):
                distance = face_recognition.face_distance([known_face], unknown_encoding)[0]
                if distance <= threshold:
                    username = known_names[i]
                    user_id_str = known_user_ids[username]
                    user_info = db.user.find_one(
                        {"_id": ObjectId(user_id_str)},
                        {"_id": 0, "username": 1, "name": 1, "department": 1, "hire_date": 1, "salary": 1}
                    ) or {}
                    return {
                        "username": user_info.get("username", "Non spécifié"),
                        "name": user_info.get("name", "Non spécifié"),
                        "department": user_info.get("department", "Non spécifié"),
                        "hire_date": user_info.get("hire_date", "Non spécifié"),
                        "salary": user_info.get("salary", "Non spécifié")
                    }
        return {"username": "Inconnu", "name": "Inconnu",
                "department": "Inconnu", "hire_date": "Inconnu", "salary": "Inconnu"}
    except Exception as e:
        logging.error(f"[FACE] Erreur identification image {file_path} : {e}")
        return {"username": "Erreur", "name": "Erreur",
                "department": "Erreur", "hire_date": "Erreur", "salary": "Erreur"}

# --- Semantic search & Groq ---
def semantic_search_reports(query: str, top_k: int = 5):
    index = init_faiss_index()
    meta = load_meta()
    if index.ntotal == 0:
        return []

    query_emb = embedding_model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        key = str(idx)
        if key in meta:
            report = meta[key]
            report_id = report.get("report_id", "inconnu")
            snippet = report.get("text_snippet", "")[:200]
            bonus = 1.2 if query.lower() in report_id.lower() else 1.0
            results.append({
                "filename": report_id,
                "snippet": snippet,
                "file_link": f"/reports/{report_id}",
                "score": float(score) * bonus
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

@app.route('/search_reports', methods=['GET', 'POST'])
def search_reports_api():
    if request.method == 'GET':
        query = request.args.get("q", "")
        if not query:
            return jsonify({"error": "Paramètre 'q' requis"}), 400
        results = semantic_search_reports(query)
        return jsonify({"results": results}), 200

    data = request.get_json() or {}
    query = data.get("query", "")
    k = data.get("k", 5)
    if not query:
        return jsonify({"error": "Query manquante"}), 400
    results = semantic_search_reports(query, top_k=k)
    return jsonify({"results": results}), 200

def validate_and_structure_response(model_response: str, context: str):
    provenance_lines = [line.strip() for line in context.splitlines() if "Fraude détectée" in line]
    dates = sorted(set(re.findall(r'\d{4}-\d{2}-\d{2}', context)))
    times = sorted(set(re.findall(r'\d{2}:\d{2}:\d{2}', context)))
    structured = {
        "response": model_response,
        "provenance": provenance_lines,
        "dates_found": dates,
        "times_found": times,
        "warnings": []
    }
    for t in re.findall(r'\d{1,2}:\d{2}', model_response):
        if all(not time.startswith(t) for time in times):
            structured["warnings"].append(f"Heure citée '{t}' absente du contexte.")
    return structured

def generate_answer_with_context_groq(prompt: str, context: str = ""):
    SYSTEM_PROMPT = """
Tu es un assistant RH interne.
Réponds toujours en français, de manière concise et naturelle,
comme dans une conversation classique.
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_message = prompt if not context.strip() else f"Contexte :\n{context}\n\nQuestion : {prompt}"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 200
    }
    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        model_response = response.json()["choices"][0]["message"]["content"].strip()
        if not model_response:
            return {"response": "Je suis désolé, je n’ai pas trouvé d’information utile."}
        return validate_and_structure_response(model_response, context)
    except httpx.HTTPError as e:
        logging.error(f"Erreur Groq HTTP: {e}")
        return {"response": "Erreur lors de l’appel à Groq.", "provenance": [], "dates_found": [], "times_found": [], "warnings": [str(e)], "contact_hr": False}
    except Exception as e:
        logging.error(f"Erreur Groq inattendue: {e}")
        return {"response": "Erreur lors de la génération de la réponse.", "provenance": [], "dates_found": [], "times_found": [], "warnings": [str(e)], "contact_hr": False}

def find_report_files_local(query: str):
    """Retourne la liste des fichiers correspondant à la query avec un download_link complet."""
    if not os.path.exists(FILES_DIR):
        return []

    query_norm = re.sub(r'[^a-z0-9]', '', query.lower())
    matched_files = []

    for fname in os.listdir(FILES_DIR):
        fname_norm = re.sub(r'[^a-z0-9]', '', fname.lower())
        if query_norm in fname_norm:
            matched_files.append({
                "filename": fname,
                "file_link": f"/reports/{fname}",
                # Génère un lien HTTP complet cliquable
                "download_link": url_for("download_file_http", filename=fname, _external=True),
                "score": 1.0
            })
    return matched_files

from datetime import datetime, timezone

@app.route("/chat_query", methods=["POST"])
def chat_query():
    """
    Endpoint principal de chat utilisateur.
    Gère :
      - Texte simple
      - Fichiers image (reconnaissance faciale)
      - Fichiers texte (PDF, DOCX, TXT)
      - Classification d'intention
      - Réponse via Groq
    """

    def _parse_conversation():
        raw = None
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            raw = request.form.get("conversation", "[]")
        else:
            raw = (request.get_json(silent=True) or {}).get("conversation", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else raw if isinstance(raw, list) else []
        except Exception:
            return []

    def _format_short_context(messages, max_msgs=8):
        trimmed = messages[-max_msgs:]
        lines = []
        for m in trimmed:
            role = m.get("role", "user")
            content = (m.get("content") or "")
            for sep in ("\n📌", "\n📅", "\n⏰", "\n⚠️"):
                content = content.split(sep)[0].strip() if content else ""
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def _extract_user_facts_from_utterance(text: str):
        facts = []
        if not text:
            return facts
        m = re.search(r"\b(?:je m'appelle|je suis)\s+([^\n,.;!?]+)", text, flags=re.I)
        if m:
            facts.append(f"Identité: {m.group(1).strip()}")
        return facts

    def _add_facts(user_id: str, facts: list[str]):
        if not facts:
            return
        doc = db.user_facts.find_one({"user_id": user_id}) or {"user_id": user_id, "facts": []}
        existing_facts = doc.get("facts", [])
        for fact in facts:
            if fact.lower().startswith("identité:"):
                existing_facts = [f for f in existing_facts if not f.lower().startswith("identité:")]
            existing_facts.append(fact)
        db.user_facts.update_one({"user_id": user_id}, {"$set": {"facts": existing_facts}}, upsert=True)

    def _get_facts(user_id: str) -> list[str]:
        doc = db.user_facts.find_one({"user_id": user_id}, {"_id": 0, "facts": 1})
        return doc.get("facts", []) if doc else []

    def _log_chat(user_id: str, role: str, content: str, intention: str | None = None):
        try:
            db.chat_memory.insert_one({
                "user_id": user_id,
                "role": role,
                "content": content,
                "intention": intention,
                "timestamp": datetime.now(timezone.utc)

            })
        except Exception as e:
            logging.error(f"[MEM] Erreur insertion chat_memory: {e}")

    # ==== Parsing input ====
    conversation = _parse_conversation()
    data_json = request.get_json(silent=True) or {}
    user_id = request.headers.get("X-User-Id") or data_json.get("userId") or "anonymous_user"

    prompt = ""
    file = None
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        prompt = (request.form.get("query") or "").strip()
        file = request.files.get("file", None)
    else:
        prompt = (data_json.get("query") or "").strip()

    if not prompt and not file:
        return jsonify({"error": "Query manquante et aucun fichier fourni"}), 400

    _log_chat(user_id, "user", prompt)

    # ==== Si fichier uploadé ====
    if file and file.filename:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        logging.info(f"[UPLOAD] Fichier reçu: {filename} ({ext})")

        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        file_path = os.path.join(reports_dir, filename)
        file.save(file_path)

        # ---- Cas image (reconnaissance) ----
        if ext in [".png", ".jpg", ".jpeg"]:
            person_info = identify_person_from_file(file_path, threshold=FACE_RECOG_THRESHOLD)
            resp = {
                "response": f"Personne reconnue : {person_info.get('name','?')} ({person_info.get('username','?')})"
                            if person_info else "Personne non reconnue",
                "details": person_info if person_info else {},
                "intention": "reconnaissance_image",
                "facts": _get_facts(user_id)
            }
            _log_chat(user_id, "assistant", resp["response"], intention="reconnaissance_image")
            return jsonify(resp)

        # ---- Cas PDF / DOCX / TXT ----
        file_text = ""
        try:
            if ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(file.stream) as pdf:
                    for page in pdf.pages:
                        file_text += page.extract_text() or ""
            elif ext == ".docx":
                import docx
                doc = docx.Document(file)
                file_text = "\n".join([p.text for p in doc.paragraphs])
            elif ext == ".txt":
                file_text = file.read().decode("utf-8", errors="ignore")
            else:
                return jsonify({"error": "Format de fichier non supporté"}), 400
        except Exception as e:
            logging.error(f"[FILE_PARSE] Erreur lecture fichier: {e}")
            return jsonify({"error": "Impossible de lire le fichier"}), 500

        enriched_prompt = f"{prompt}\n\nContenu du fichier {filename}:\n{file_text[:3000]}..."
        answer = generate_answer_with_context_groq(enriched_prompt)
        assistant_text = answer.get("response", "")

        _log_chat(user_id, "assistant", assistant_text, intention="analyse_fichier")
        return jsonify({
            "response": assistant_text or "Pas de réponse.",
            "provenance": answer.get("provenance", []),
            "dates_found": answer.get("dates_found", []),
            "times_found": answer.get("times_found", []),
            "warnings": answer.get("warnings", []),
            "intention": "analyse_fichier",
            "facts": _get_facts(user_id)
        })

    # ==== Sinon, prompt texte classique ====
    intention = "autre"
    intent_data = {}
    try:
        classification_resp = httpx.post("http://localhost:5010/classify_intent", json={"query": prompt}, timeout=30)
        classification_resp.raise_for_status()
        intent_data = classification_resp.json()
        intention = intent_data.get("intention", "autre")
    except Exception as e:
        logging.error(f"[INTENT] Erreur classification: {e}")
        intent_data = {"response": "Erreur classification."}

    short_context_block = _format_short_context(conversation, max_msgs=8)
    facts = _get_facts(user_id)
    facts_text = "; ".join(facts) if facts else "Aucun fait enregistré."
    enriched_prompt = (
        f"Faits connus (mémoire longue) sur [{user_id}] :\n{facts_text}\n\n"
        f"Contexte récent (mémoire courte) :\n{short_context_block or 'Aucun échange récent.'}\n\n"
        f"Question actuelle :\n{prompt}"
    )

    if intention == "demande_rapport":
        matched_reports = find_report_files_local(prompt)
        if matched_reports:
            resp_text = "Voici le(s) document(s) correspondant à votre demande."
            return jsonify({
                "response": resp_text,
                "matched_reports": matched_reports,  # <-- inclut filename + download_link
                "intention": intention
            })
        else:
            return jsonify({
                "response": "Désolé, je n'ai trouvé aucun fichier correspondant.",
                "matched_reports": [],
                "intention": intention
            })

    if intention == "get_file":
        filename = intent_data.get("filename")
        if not filename:
            return jsonify({"response": "Je n’ai pas compris quel fichier vous voulez.", "intention": intention})

        file_path = os.path.join(os.getcwd(), "reports", filename)
        if os.path.exists(file_path):
            download_link = url_for("download_file_http", filename=filename, _external=True)
            return jsonify({"response": f"Voici le fichier demandé : {filename}",
                            "download_link": download_link,
                            "intention": "get_file"})
        else:
            return jsonify({"response": f"Le fichier {filename} est introuvable dans /reports.",
                            "intention": "get_file"})

    answer = generate_answer_with_context_groq(enriched_prompt)
    assistant_text = answer.get("response", "")
    _log_chat(user_id, "assistant", assistant_text, intention=intention)
    _add_facts(user_id, _extract_user_facts_from_utterance(prompt))

    return jsonify({
        "response": assistant_text or "Pas de réponse.",
        "provenance": answer.get("provenance", []),
        "dates_found": answer.get("dates_found", []),
        "times_found": answer.get("times_found", []),
        "warnings": answer.get("warnings", []),
        "intention": intention,
        "facts": _get_facts(user_id)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5010)