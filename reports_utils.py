import os
import json
import pandas as pd
import fitz  # PyMuPDF pour PDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import logging
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import pytesseract
import docx2txt
# Configuration du binaire Tesseract pour Windows si non détecté
if os.name == "nt":
    try:
        default_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", default_tesseract)
    except Exception:
        pass
# Initialisation modèle embedding (tu peux changer le modèle)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS paths
INDEX_PATH = "reports/faiss_index.bin"
META_PATH = "reports/meta.json"

def mongo_connection():
    try:
        client = MongoClient("mongodb+srv://stage:stage@cluster0.gbm1c.mongodb.net/stage?retryWrites=true&w=majority")
        return client["stage"]
    except Exception as e:
        logging.error(f"Erreur de connexion à MongoDB: {e}")
        return None

def extract_text_from_image(image_path: str) -> str:
    """OCR image -> texte (français si dispo)."""
    try:
        img = Image.open(image_path)
        # Tente FR puis fallback EN si FR non installé
        try:
            text = pytesseract.image_to_string(img, lang="fra")
        except Exception:
            text = pytesseract.image_to_string(img)
        return text or ""
    except Exception as e:
        logging.error(f"Erreur extraction image {image_path} : {e}")
        return ""




def extract_text_from_docx(docx_path: str) -> str:
    """DOCX -> texte"""
    try:
        text = docx2txt.process(docx_path)
        return text or ""
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {docx_path} : {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {pdf_path} : {e}")
        return ""

def extract_text_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        return df.to_csv(index=False)
    except Exception as e:
        logging.error(f"Erreur extraction Excel {excel_path} : {e}")
        return ""

def init_faiss_index(dim=384):
    if not os.path.exists("reports"):
        os.makedirs("reports")
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        logging.info("Index FAISS chargé")
    else:
        index = faiss.IndexFlatL2(dim)
        logging.info("Nouveau index FAISS créé")
    return index

def index_additional_docs(folder_path="knowledge_base"):
    """
    Parcourt un dossier et indexe uniquement les fichiers non encore présents
    dans la base FAISS, pour éviter les doublons.
    """
    supported_ext = [".pdf", ".xlsx", ".xls", ".txt", ".docx", ".png", ".jpg", ".jpeg"]

    if not os.path.exists(folder_path):
        logging.warning(f"Dossier {folder_path} introuvable — création en cours.")
        os.makedirs(folder_path)
        return  # Rien à indexer la première fois

    meta = load_meta()
    index = init_faiss_index()

    # Récupérer la liste des fichiers déjà indexés
    fichiers_indexes = {v["report_id"] for v in meta.values()}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in supported_ext:
            logging.warning(f"Format non supporté : {file_name}")
            continue

        if file_name in fichiers_indexes:
            logging.info(f"⏩ Fichier déjà indexé, ignoré : {file_name}")
            continue

        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            text = extract_text_from_excel(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".docx":
            text = extract_text_from_docx(file_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = extract_text_from_image(file_path)
        else:
            text = ""

        if text.strip():
            logging.info(f"✅ Indexation du fichier : {file_name}")
            index_report(file_name, text, index, meta)
        else:
            logging.warning(f"⚠️ Fichier vide ou illisible : {file_name}")

def save_index(index):
    faiss.write_index(index, INDEX_PATH)
    logging.info("Index FAISS sauvegardé")

def save_meta(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info("Meta sauvegardé")

def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def index_report(report_id, text, index, meta):
    embedding = embedding_model.encode([text])
    index.add(np.array(embedding).astype('float32'))
    meta[str(len(meta))] = {"report_id": report_id, "text_snippet": text[:200]}
    save_index(index)
    save_meta(meta)

def generate_daily_reports():
    logging.info("📝 Début génération rapports quotidiens")

    db = mongo_connection()
    if db is None:
        logging.error("Erreur base données - rapports")
        return

    date_cible = datetime.now().strftime('%Y-%m-%d')
    date_debut = datetime.strptime(date_cible, "%Y-%m-%d")
    date_fin = date_debut + timedelta(days=1)

    # Extraction Pointages
    pointages_cursor = db.pointage.find({
        "date_pointage": {"$gte": date_debut, "$lt": date_fin}
    })
    pointages = list(pointages_cursor)

    if not pointages:
        logging.warning(f"Aucun pointage trouvé pour la date {date_cible}")
        df_pointage = pd.DataFrame()
    else:
        rows = []
        for p in pointages:
            rows.append({
                "date": p.get("date_pointage").strftime("%Y-%m-%d") if p.get("date_pointage") else date_cible,
                "username": p.get("username", "Inconnu"),
                "role": p.get("role", "Inconnu"),
                "arrivees": len(p.get("arrivees", [])),
                "departs": len(p.get("departs", []))
            })
        df_pointage = pd.DataFrame(rows)

    # Génération Excel
    excel_path = f"reports/pointage_{date_cible}.xlsx"
    if not df_pointage.empty:
        df_pointage.to_excel(excel_path, index=False)
        logging.info(f"✅ Rapport Excel pointage généré : {excel_path}")
    else:
        logging.info("Pas de rapport Excel pointage généré (pas de données)")

    # Extraction Fraudes
    fraudes_cursor = db.fraude.find({
        "date": {"$gte": date_debut, "$lt": date_fin}
    }).sort("date", 1)
    fraudes = list(fraudes_cursor)

    # Génération PDF résumé fraudes
    pdf_path = f"reports/fraude_{date_cible}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Rapport Fraudes - Date : {date_cible}")

    c.setFont("Helvetica", 12)
    y = height - 80
    if not fraudes:
        c.drawString(50, y, "Aucune fraude détectée.")
    else:
        for f in fraudes:
            line = f"{f.get('date').strftime('%H:%M:%S')} - {f.get('username')} - {f.get('message')}"
            c.drawString(50, y, line)
            y -= 20
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()
    logging.info(f"✅ Rapport PDF fraude généré : {pdf_path}")

def index_daily_reports():
    logging.info("🗂️ Indexation des rapports quotidiens dans FAISS")

    date_cible = datetime.now().strftime('%Y-%m-%d')
    excel_path = f"reports/pointage_{date_cible}.xlsx"
    pdf_path = f"reports/fraude_{date_cible}.pdf"

    meta = load_meta()
    index = init_faiss_index()

    if os.path.exists(excel_path):
        text_excel = extract_text_from_excel(excel_path)
        index_report(f"pointage_{date_cible}", text_excel, index, meta)
        logging.info(f"Indexé Excel: {excel_path}")

    if os.path.exists(pdf_path):
        text_pdf = extract_text_from_pdf(pdf_path)
        index_report(f"fraude_{date_cible}", text_pdf, index, meta)
        logging.info(f"Indexé PDF: {pdf_path}")

def search_reports(query, k=3):
    index = init_faiss_index()
    meta = load_meta()

    if index.ntotal == 0:
        logging.warning("Index FAISS vide")
        return []

    query_emb = embedding_model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), k)

    results = []
    for idx in I[0]:
        key = str(idx)
        if key in meta:
            results.append(meta[key])
    return results
def index_all_reports():
    logging.info("🗂️ Indexation de tous les rapports existants")

    reports_dir = os.path.join(os.getcwd(), "reports")
    if not os.path.exists(reports_dir):
        logging.warning("Dossier reports inexistant")
        return

    index = init_faiss_index()
    meta = load_meta()

    for filename in os.listdir(reports_dir):
        file_path = os.path.join(reports_dir, filename)
        if os.path.isfile(file_path):
            report_id = os.path.splitext(filename)[0]

            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith((".xls", ".xlsx")):
                text = extract_text_from_excel(file_path)
            elif filename.lower().endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                text = extract_text_from_image(file_path)
            else:
                logging.info(f"Fichier ignoré (type non supporté) : {filename}")
                continue

            index_report(report_id, text, index, meta)
            logging.info(f"Indexé : {filename}")

    logging.info("✅ Indexation complète terminée")
