# fraude_detector.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, logging, sys, gc
import cv2, numpy as np
from typing import List, Tuple, Dict
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

# ========= CONFIG =========
PORT = 5001
DEBUG = True
MAX_WORKERS = 6
TARGET_FRAMES_FOR_SAMPLING = 60
TMP_DIR = "temp_videos"
os.makedirs(TMP_DIR, exist_ok=True)

# Seuils
SEUIL_LIVENESS = 0.15
SEUIL_PHONE_CONF = 0.15
SEUIL_SCREEN_PROBA = 0.60

# Screen features
FRAME_STEP_SCREEN = 5
IMG_SIZE_SCREEN = (224, 224)
FEATURE_SIZE = 1280*2 + 4  # correspond au vecteur CNN + 4 z-features

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO if DEBUG else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========= Flask =========
app = Flask(__name__)
CORS(app)

# ========= YOLOv8 finetuné =========
try:
    from ultralytics import YOLO
    YOLO_MODEL_PATH = "C:\\Users\\asus\\avec_fraude\\runs\\detect\\yolo_phone_finetune_aug\\weights\\best.pt"
    logging.info("Chargement YOLOv8 finetuné (téléphones)...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    logging.info(f"[DEBUG] Classes YOLO chargées: {yolo_model.names}")
    yolo_ok = True
except Exception as e:
    logging.error(f"YOLOv8 finetuné introuvable/indisponible: {e}")
    yolo_model = None
    yolo_ok = False

# ========= Liveness MobileNet =========
class LivenessWrapper:
    def __init__(self):
        self.ok = False
        self.net = None
        try:
            REPO_DIR = os.path.join(os.getcwd(), "face-antispoofing-using-mobileNet")
            if os.path.isdir(REPO_DIR) and REPO_DIR not in sys.path:
                sys.path.insert(0, REPO_DIR)
            weights_path = os.path.join(REPO_DIR, "pretrained_weights", "mobilenet_v1.pkl")
            import torch
            from layers import MobileNetV1
            self.net = MobileNetV1(num_classes=2)
            state = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state:
                self.net.load_state_dict(state["state_dict"])
            else:
                self.net.load_state_dict(state)
            self.net.eval()
            self.ok = True
            logging.info("Liveness MobileNet chargé.")
        except Exception as e:
            self.ok = False
            self.net = None
            logging.warning(f"Liveness MobileNet non disponible, fallback heuristique: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0+side, x0:x0+side]
        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - 0.5) / 0.5
        crop = np.transpose(crop, (2, 0, 1))
        return crop

    def predict_score(self, frame: np.ndarray) -> float:
        try:
            if self.ok and self.net is not None:
                import torch
                x = self._preprocess(frame)
                x = torch.from_numpy(x).unsqueeze(0)
                with torch.no_grad():
                    logits = self.net(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                return float(probs[1])
            # Fallback heuristique
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            return max(0.0, min(var / 500.0, 1.0))
        except Exception as e:
            logging.warning(f"Liveness prediction error: {e}")
            return 0.0

liveness_model = LivenessWrapper()

# ========= Screen recording XGB + CNN =========
SCR_MODEL_PATH = "cnn_xgb_finetuned_model.pkl"
screen_model = None
screen_model_ok = False
cnn_model_screen = None

try:
    screen_model = joblib.load(SCR_MODEL_PATH)
    screen_model_ok = True
    logging.info("Modèle XGBoost (screen recording) chargé.")
except Exception as e:
    logging.warning(f"Modèle screen recording indisponible: {e}")

try:
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg', input_shape=(224,224,3))
    x = Dropout(0.5)(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    full_cnn_model = Model(inputs=base_model.input, outputs=output)
    cnn_model_screen = Model(inputs=full_cnn_model.input, outputs=full_cnn_model.layers[-2].output)
    logging.info("CNN MobileNetV2 fine-tuné aligné avec l’entraînement.")
except Exception as e:
    logging.warning(f"Impossible de charger CNN fine-tuné: {e}")
    cnn_model_screen = None

def extract_video_features_screen(video_path: str) -> np.ndarray:
    if cnn_model_screen is None or not screen_model_ok:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        ret, frame = cap.read()
        while ret:
            if idx % FRAME_STEP_SCREEN == 0:
                frames.append(cv2.resize(frame, IMG_SIZE_SCREEN))
            ret, frame = cap.read()
            idx += 1
        cap.release()
        if not frames:
            return np.zeros(FEATURE_SIZE, dtype=np.float32)
        frames_array = preprocess_input(np.array(frames, dtype=np.float32))
        cnn_feats = cnn_model_screen.predict(frames_array, verbose=0)
        cnn_mean = np.mean(cnn_feats, axis=0)
        cnn_std  = np.std(cnn_feats, axis=0)
        feature_vector = np.concatenate([cnn_mean, cnn_std, [0.0,0.0,0.0,0.0]])
        if feature_vector.shape[0] < FEATURE_SIZE:
            feature_vector = np.pad(feature_vector, (0, FEATURE_SIZE - feature_vector.shape[0]), mode='constant')
        elif feature_vector.shape[0] > FEATURE_SIZE:
            feature_vector = feature_vector[:FEATURE_SIZE]
        return feature_vector.astype(np.float32)
    except Exception as e:
        logging.warning(f"extract_video_features_screen error: {e}")
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

def screen_model_proba(video_path: str) -> float:
    if not screen_model_ok or cnn_model_screen is None:
        return 0.0
    try:
        feats = extract_video_features_screen(video_path).reshape(1,-1)
        proba = float(screen_model.predict_proba(feats)[0,1])
        return proba
    except Exception as e:
        logging.warning(f"Screen model predict_proba error: {e}")
        return 0.0

# ========= Utilitaires vidéo =========
def safe_video_meta(path: str) -> Tuple[float,int,int,int]:
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return 0.0,0,0,0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        dur = (total/fps) if fps>0 else 0.0
        return dur,total,width,height
    except Exception as e:
        logging.warning(f"safe_video_meta error: {e}")
        return 0.0,0,0,0

def sample_frames(path: str, target_frames: int) -> List[np.ndarray]:
    frames = []
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total == 0: cap.release(); return frames
        step = max(1, total // target_frames)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
    except Exception as e:
        logging.warning(f"sample_frames error: {e}")
    return frames

# ========= Détections par frame =========
def detect_liveness_frame(frame: np.ndarray) -> Tuple[bool,float,List[str]]:
    try:
        prob_real = liveness_model.predict_score(frame)
        fraude = prob_real < SEUIL_LIVENESS
        reasons = [f"Liveness faible ({prob_real:.2f})"] if fraude else []
        return fraude,float(prob_real),reasons
    except Exception as e:
        logging.warning(f"Liveness frame error: {e}")
        return False,0.0,[]

def detect_phone_frame(frame: np.ndarray) -> Tuple[bool,float,List[str]]:
    if not yolo_ok: return False,0.0,[]
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(source=frame_rgb, verbose=False)
        fraude = False
        best_conf = 0.0
        reasons = []
        for det in results[0].boxes:
            cls_id = int(det.cls)
            conf = float(det.conf)
            cls_name = yolo_model.names.get(cls_id, str(cls_id))
            if cls_name == "phone" and conf >= SEUIL_PHONE_CONF:
                fraude = True
                best_conf = max(best_conf, conf)
                reasons.append("Téléphone détecté")
        return fraude,float(best_conf),reasons
    except Exception as e:
        logging.warning(f"YOLO erreur frame: {e}")
        return False,0.0,[]

def detect_screen_recording(video_path: str) -> Tuple[bool,float,List[str]]:
    try:
        proba = screen_model_proba(video_path)
        fraude = proba > SEUIL_SCREEN_PROBA
        reasons = ["Screen recording suspect"] if fraude else []
        return fraude,float(proba),reasons
    except Exception as e:
        logging.warning(f"Screen recording error: {e}")
        return False,0.0,[]

# ========= Analyse principale =========
FRAME_SUSPECT_DIR = "frames_suspectes"
os.makedirs(FRAME_SUSPECT_DIR, exist_ok=True)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_on_screen(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    screen_detected = False
    for (x,y,w,h) in faces:
        roi = frame[y:y+h,x:x+w]
        edges = cv2.Canny(roi,100,200)
        contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
            if len(approx)==4 and cv2.contourArea(cnt)>500:
                screen_detected = True
                break
        if screen_detected: break
    return screen_detected

def analyser_video(video_path: str) -> Dict:
    dur,total,w,h = safe_video_meta(video_path)
    result: Dict = {
        "video_path": video_path,
        "duration": float(dur),
        "frames": int(total),
        "width": int(w),
        "height": int(h),
        "fraude": False,
        "reasons": [],
        "scores": {}
    }
    frames = sample_frames(video_path, TARGET_FRAMES_FOR_SAMPLING)
    if frames:
        gris_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        luminosites = [float(np.mean(f)) for f in gris_frames]
        contrastes = [float(np.std(f)) for f in gris_frames]
        luminosite_moyenne = float(np.mean(luminosites))
        contraste_moyen = float(np.mean(contrastes))
    else:
        luminosite_moyenne = 0.0
        contraste_moyen = 0.0
    SEUIL_MIN, SEUIL_MAX = 0.10,0.20
    BUFFER_LIVENESS=0.0
    SEUIL_LIVENESS_ADAPT = SEUIL_MIN + (luminosite_moyenne/255)*(SEUIL_MAX-SEUIL_MIN) - BUFFER_LIVENESS
    if contraste_moyen<30.0: SEUIL_LIVENESS_ADAPT*=0.95
    print(f"[DEBUG] Lum: {luminosite_moyenne:.2f}, Contraste: {contraste_moyen:.2f}, Seuil Liveness: {SEUIL_LIVENESS_ADAPT:.3f}")
    liveness_scores, phone_scores = [],[]
    for i,f in enumerate(frames):
        lf,ls,lr = detect_liveness_frame(f)
        pf,ps,pr = detect_phone_frame(f)
        screen_face_detected = detect_face_on_screen(f)
        liveness_scores.append(float(ls))
        phone_scores.append(float(ps))
        result["reasons"] += lr + pr
        if pf:
            frame_file = os.path.join(FRAME_SUSPECT_DIR,f"{os.path.basename(video_path)}_phone_frame_{i+1}.jpg")
            cv2.imwrite(frame_file,f)
        if screen_face_detected:
            frame_file = os.path.join(FRAME_SUSPECT_DIR,f"{os.path.basename(video_path)}_screen_frame_{i+1}.jpg")
            cv2.imwrite(frame_file,f)
            result["reasons"].append("Visage affiché sur un écran")
            result["fraude"] = True
    liveness_mean = float(np.mean(liveness_scores)) if liveness_scores else 0.0
    phone_mean = float(np.mean(phone_scores)) if phone_scores else 0.0
    result["scores"]["liveness"]=liveness_mean
    result["scores"]["phone"]=phone_mean
    if any(r in result["reasons"] for r in ["Téléphone détecté","Visage affiché sur un écran"]):
        result["fraude"]=True
    elif liveness_mean<SEUIL_LIVENESS_ADAPT:
        result["fraude"]=True
    scr_f,scr_p,scr_r = detect_screen_recording(video_path)
    result["scores"]["screen"]=float(scr_p)
    if scr_f:
        result["fraude"]=True
        result["reasons"]+=scr_r
    result["seuil_liveness_adaptatif"]=float(SEUIL_LIVENESS_ADAPT)
    return result

# ========= Routes Flask =========
@app.route("/detect_fraude",methods=["POST"])
def detect_fraude():
    if 'video' not in request.files:
        return jsonify({"error":"Aucune vidéo envoyée"}),400
    f=request.files['video']
    vid_id=str(uuid.uuid4())
    vid_path=os.path.join(TMP_DIR,f"{vid_id}.mp4")
    f.save(vid_path)
    try:
        res = analyser_video(vid_path)
    finally:
        try: os.remove(vid_path)
        except: pass
        gc.collect()
    return jsonify(res)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
