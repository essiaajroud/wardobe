"""
FastAPI Backend for AI Outfit Recommender
Adjusted to match the Jupyter Notebook implementation exactly

Setup:
    pip install fastapi uvicorn python-multipart pillow tensorflow opencv-python 
    pip install webcolors scikit-learn xgboost requests numpy pandas

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import shutil
import uuid
from datetime import datetime
import pickle
import requests
import random
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import cv2
import webcolors
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import itertools
from sklearn.metrics import mean_squared_error

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="AI Outfit Recommender API",
    description="Context-aware outfit recommendations with RL",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CONFIGURATION
# ============================================

UPLOAD_DIR = "./uploads/wardrobe"
MODEL_GENERAL_PATH = "./models/fashion_main_categories.keras"
MODEL_DRESS_PATH = "./models/fashion_dress_styles.keras"
FEEDBACK_PATH = "./data/outfit_feedback.pkl"
XGB_MODEL_PATH = "./data/xgb_model.pkl"
ENCODER_PATH = "./data/encoder.pkl"
SCALER_PATH = "./data/scaler.pkl"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./data", exist_ok=True)

OCCASIONS = ["casual", "work", "party", "wedding"]

CLASSES_GENERAL = ["Coat", "Dress", "Jeans", "Skirt", "Top"]
CLASSES_DRESS = ["Dress_Casual", "Dress_Wedding"]

COLOR_GROUPS = {
    "Black": "neutral", "White": "neutral", "Gray": "neutral", "Beige": "neutral",
    "Blue": "cool", "Green": "cool", "Purple": "cool",
    "Red": "warm", "Orange": "warm", "Yellow": "warm", "Pink": "warm", "Brown": "warm"
}

COLOR_COMPAT = {
    ("neutral", "neutral"): 1.0,
    ("neutral", "cool"): 0.9,
    ("neutral", "warm"): 0.9,
    ("cool", "neutral"): 0.9,
    ("warm", "neutral"): 0.9,
    ("cool", "cool"): 0.8,
    ("warm", "warm"): 0.7,
    ("cool", "warm"): 0.4,
    ("warm", "cool"): 0.4
}

VALID_STRUCTURES = [
    ["Top", "Jeans"],
    ["Top", "Skirt"],
    ["Top", "Jeans", "Coat"],
    ["Top", "Skirt", "Coat"],
    ["Dress"],
    ["Dress", "Coat"]
]

def get_closest_color(requested_colour):
    min_colours = {}
    
    # On prépare un dictionnaire de couleurs (Nom -> RGB)
    color_map = {}
    
    try:
        # Méthode pour webcolors 2.0+ (Utilise le registre CSS3)
        for name in webcolors.names(spec="css3"):
            color_map[name] = webcolors.name_to_rgb(name)
    except (AttributeError, ValueError):
        try:
            # Fallback pour les anciennes versions (CSS3_NAMES_TO_HEX)
            for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items():
                color_map[name] = webcolors.hex_to_rgb(hex_code)
        except AttributeError:
            # Dernier recours : Couleurs de base si la bibliothèque est mal installée
            color_map = {
                "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
                "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
                "gray": (128, 128, 128), "orange": (255, 165, 0), "pink": (255, 192, 203),
                "purple": (128, 0, 128), "brown": (165, 42, 42)
            }

    # Calcul de la distance pour chaque couleur du dictionnaire
    for name, rgb_values in color_map.items():
        r_c, g_c, b_c = rgb_values
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
        
    return min_colours[min(min_colours.keys())]



# ============================================
# GLOBAL STATE
# ============================================

class AppState:
    def __init__(self):
        self.wardrobe = []
        self.cnn_general = None
        self.cnn_styles = None
        self.base_model = None
        self.xgb_model = None
        self.encoder = None
        self.scaler = None
        self.num_features = ["temp", "color_score", "warmth", "breath", "has_dress", "has_coat"]
        self.cat_features = ["occasion"]
        self.weather_service = None
        self.feedback_store = None
        self.rl_recommender = None
        self.is_initialized = False

state = AppState()

# ============================================
# WEATHER SERVICE (from notebook)
# ============================================

class WeatherService:
    """Fetch real-time weather data"""

    def __init__(self, api_key=None):
        self.api_key = api_key or "140780ba37e77f0db03188e8b323d7b1"
        # OpenWeatherMap API endpoint
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def _get_location_from_ip(self):
        try:
            response = requests.get("https://ipapi.co/json/", timeout=5)
            data = response.json()
            return data.get('city', 'Sousse'), data.get('country_code', 'TN')
        except:
            return "Sousse", "TN"

    def get_weather(self, city=None, country_code=None):
        if not city: 
            city, country_code = self._get_location_from_ip()
        
        try:
            url = f"{self.base_url}?q={city},{country_code}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'temperature': round(data['main']['temp'], 1),
                    'feels_like': round(data['main']['feels_like'], 1),
                    'weather': data['weather'][0]['main'],
                    'description': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'city': city,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return self._get_fallback_weather(city)
        except:
            return self._get_fallback_weather(city)

    def _get_fallback_weather(self, city="Sousse"):
        temp = random.randint(12, 18) # Température réaliste pour Janvier en Tunisie
        return {
            'temperature': float(temp),
            'feels_like': float(temp),
            'weather': 'Clouds',
            'description': 'overcast clouds',
            'humidity': 70,
            'wind_speed': 5.0,
            'city': city,
            'timestamp': datetime.now().isoformat()
        }

    

# ============================================
# FEEDBACK STORE (from notebook)
# ============================================

class OutfitFeedbackStore:
    """Store and manage user feedback for RL"""

    def __init__(self, storage_path="./data/outfit_feedback.pkl"):
        self.storage_path = storage_path
        self.feedback_history = self._load_feedback()
        self.total_recommendations = len(self.feedback_history)
        self.positive_feedback = sum(1 for f in self.feedback_history if f['feedback_score'] > 0)
        self.negative_feedback = sum(1 for f in self.feedback_history if f['feedback_score'] < 0)

    def _load_feedback(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    print(f"📚 Loaded {len(data)} feedback entries")
                    return data
            except Exception as e:
                print(f"⚠️ Error loading feedback: {e}")
        return []

    def _save_feedback(self):
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.feedback_history, f)
        except Exception as e:
            print(f"❌ Error saving feedback: {e}")

    def add_feedback(self, outfit, temperature, occasion, feedback_score, reason=None):
        """Record user feedback (1=like, 0=neutral, -1=dislike)"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'outfit': [{'type': item['type'], 'color': item['color']} for item in outfit],
            'temperature': temperature,
            'occasion': occasion,
            'feedback_score': feedback_score,
            'reason': reason
        }

        self.feedback_history.append(feedback_entry)
        self.total_recommendations += 1

        if feedback_score > 0:
            self.positive_feedback += 1
            emoji = "👍"
        elif feedback_score < 0:
            self.negative_feedback += 1
            emoji = "👎"
        else:
            emoji = "😐"

        self._save_feedback()

        outfit_str = " + ".join([item['type'] for item in outfit])
        print(f"{emoji} Feedback recorded: {outfit_str}")

    def get_statistics(self):
        if self.total_recommendations == 0:
            return {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'positive_rate': 0.0,
            'negative_rate': 0.0,
            'neutral_rate': 0.0
        }

        neutral = self.total_recommendations - self.positive_feedback - self.negative_feedback

        return {
            'total': self.total_recommendations,
            'positive': self.positive_feedback,
            'negative': self.negative_feedback,
            'neutral': neutral,
            'positive_rate': round(self.positive_feedback / self.total_recommendations * 100, 1),
            'negative_rate': round(self.negative_feedback / self.total_recommendations * 100, 1),
            'neutral_rate': round(neutral / self.total_recommendations * 100, 1)
        }

    def get_preferred_combinations(self, min_score=0, min_count=1):
        """Analyze feedback to find preferred outfit combinations"""
        preferences = {}

        for entry in self.feedback_history:
            if entry['feedback_score'] < min_score:
                continue

            outfit_key = tuple(sorted([item['type'] for item in entry['outfit']]))
            occasion = entry['occasion']
            temp_range = self._get_temp_range(entry['temperature'])

            key = (occasion, temp_range, outfit_key)

            if key not in preferences:
                preferences[key] = {'count': 0, 'total_score': 0, 'outfit_types': outfit_key}

            preferences[key]['count'] += 1
            preferences[key]['total_score'] += entry['feedback_score']

        return {
            k: {**v, 'avg_score': v['total_score'] / v['count']}
            for k, v in preferences.items()
            if v['count'] >= min_count
        }

    def _get_temp_range(self, temperature):
        if temperature < 10:
            return 'cold'
        elif temperature < 20:
            return 'cool'
        elif temperature < 28:
            return 'warm'
        else:
            return 'hot'

# ============================================
# RL RECOMMENDER (from notebook)
# ============================================

class RLRecommender:
    """Enhanced recommender with RL"""

    def __init__(self, base_model, feedback_store):
        self.base_model = base_model
        self.feedback_store = feedback_store
        self.exploration_rate = 0.15

    def recommend_with_rl(self, wardrobe, encoder, scaler, num_features, cat_features,
                      temperature, occasion, top_k=3):
        """
        Génère les meilleures recommandations avec :
        - Filtres logiques stricts (No Jeans in Weddings, etc.)
        - Ajustement basé sur le feedback (RL)
        - Diversité (Shuffle) pour éviter la répétition
        """

        all_outfits = generate_outfits(wardrobe)
        scored_outfits = []

        for outfit in all_outfits:
            # --- FILTRES DE SÉCURITÉ LOGIQUE (Hard Constraints) ---
            types = [item["type"] for item in outfit]
            
            # 1. Règles pour le Mariage (Wedding)
            if occasion == "wedding":
                # On élimine immédiatement les jeans et les robes casual du mariage
                if "Jeans" in types or "Dress_Casual" in types:
                    continue 
            
            # 2. Règles pour le Quotidien/Travail
            else:
                # On élimine la robe de mariée si ce n'est pas un mariage
                if "Dress_Wedding" in types:
                    continue

            # --- EXTRACTION DES FEATURES ---
            outfit_feats = outfit_to_vector(outfit, state.base_model)
            if outfit_feats is None:
                continue

            # --- PRÉDICTION XGBOOST (Base Score) ---
            numerical_values = [
                temperature,
                outfit_feats["color_score"],
                outfit_feats["warmth"],
                outfit_feats["breath"],
                outfit_feats["has_dress"],
                outfit_feats["has_coat"]
            ]

            # Transformation des données pour le modèle
            numerical_df = pd.DataFrame([numerical_values], columns=num_features)
            scaled_num = scaler.transform(numerical_df)
            encoded_cat = encoder.transform(pd.DataFrame([{"occasion": occasion}])[cat_features])
            X_predict = np.hstack([scaled_num, encoded_cat])

            # Calcul du score via le Régresseur
            base_score = float(self.base_model.predict(X_predict)[0])
            base_score = max(0.0, min(1.0, base_score)) 

            # --- AJUSTEMENTS ET PRIORITÉS ---
            
            # 1. Bonus de priorité pour les robes de mariage lors d'un mariage
            if occasion == "wedding" and "Dress_Wedding" in types:
                base_score += 0.5 
                
            # 2. Ajustement RL (Feedback utilisateur)
            # _get_rl_adjustment doit renvoyer une valeur négative si Dislike (-0.8) 
            # ou positive si Like (+0.2)
            rl_adjustment = self._get_rl_adjustment(outfit, temperature, occasion)
            
            # Calcul du score final (Score de base * Multiplicateur de feedback)
            # On limite le score final pour garder une cohérence
            final_score = min(base_score * (1 + rl_adjustment), 1.5)

            scored_outfits.append({
                'outfit': outfit,
                'base_score': base_score,
                'rl_adjustment': rl_adjustment,
                'final_score': final_score
            })

        # Si aucune tenue ne correspond aux filtres
        if not scored_outfits:
            return []

        # --- TRI ET DIVERSITÉ (Shuffle) ---
        
        # 1. On trie d'abord par score pour identifier les meilleurs candidats
        scored_outfits.sort(key=lambda x: x['final_score'], reverse=True)

        # 2. Pour éviter de voir toujours les 3 mêmes choix :
        # On prend les 10 meilleurs candidats (Top 10)
        top_pool = scored_outfits[:min(10, len(scored_outfits))]
        
        # 3. On mélange ce Top 10 aléatoirement
        random.shuffle(top_pool)

        # 4. On trie à nouveau légèrement le pool mélangé pour s'assurer 
        # que les tenues détestées (RL négatif) restent quand même en bas.
        top_pool.sort(key=lambda x: x['final_score'], reverse=True)

        # Retourne les Top K résultats
        return top_pool[:top_k]

    def _get_rl_adjustment(self, outfit, temperature, occasion):
        outfit_types = tuple(sorted([item['type'] for item in outfit]))
        temp_range = self.feedback_store._get_temp_range(temperature)
        preferences = self.feedback_store.get_preferred_combinations(min_score=-1, min_count=1)
        key = (occasion, temp_range, outfit_types)

        if key in preferences:
            pref = preferences[key]
            avg_score = pref['avg_score'] # Si Dislike, c'est -1
            count = pref['count']
            confidence = min(count / 3, 1.0) # Confiance plus rapide (3 votes suffisent)
            
            # ✅ AUGMENTATION DE L'IMPACT : de 0.15 à 0.8
            adjustment = avg_score * 0.8 * confidence 
            return adjustment
        return 0.0

# ============================================
# CORE FUNCTIONS (from notebook)
# ============================================

def extract_cnn_features(img_path, base_model):
    """Extract CNN features from image"""
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    feat = base_model.predict(arr, verbose=0)
    return feat[0]

def compute_thermal_scores(cnn_feat):
    """Compute warmth and breathability from CNN features"""
    warmth = np.mean(cnn_feat[:200])
    breath = np.mean(cnn_feat[200:400])
    warmth = np.clip(warmth, 0, 1)
    breath = np.clip(breath, 0, 1)
    return warmth, breath

def predict_clothing_cascading(img_path):
    # 1. Prédiction Générale
    img = image.load_img(img_path, target_size=(224, 224))
    arr = preprocess_input(image.img_to_array(img))
    arr = np.expand_dims(arr, axis=0)
    
    preds_gen = state.cnn_general.predict(arr, verbose=0)
    label = CLASSES_GENERAL[np.argmax(preds_gen)]
    confidence = np.max(preds_gen)

    # 2. Logique de Cascade : Si c'est une robe, on demande à l'expert
    if label == "Dress":
        preds_style = state.cnn_styles.predict(arr, verbose=0)
        label = CLASSES_DRESS[np.argmax(preds_style)]
        confidence = np.max(preds_style) 

    return label, confidence

def detect_color(img_path):
    """Detect dominant color"""
    img = cv2.imread(img_path)
    if img is None: return "Unknown"
    
    img = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(pixels)
    color = kmeans.cluster_centers_[0].astype(int)
    
    try:
        # Essaye la correspondance exacte d'abord
        return webcolors.rgb_to_name(tuple(color)).capitalize()
    except ValueError:
        # ✅ FIX : Si pas de correspondance exacte, cherche la plus proche
        return get_closest_color(color).capitalize()

def color_score(outfit):
    """Calculate color compatibility score"""
    groups = [COLOR_GROUPS.get(i["color"], "neutral") for i in outfit]
    scores = [COLOR_COMPAT.get((a, b), COLOR_COMPAT.get((b, a), 0.5)) 
              for a, b in itertools.combinations(groups, 2)]
    return np.mean(scores) if scores else 0.5

def generate_outfits(wardrobe):
    """Generate all valid outfit combinations"""
    outfits = []
    for struct in VALID_STRUCTURES:
        items = []
        for t in struct:
            cand = [w for w in wardrobe if w["type"] == t]
            if not cand:
                break
            items.append(cand)
        if len(items) == len(struct):
            for combo in itertools.product(*items):
                outfits.append(list(combo))
    return outfits

def smart_label(outfit, temperature, occasion):
    """L'intelligence de l'IA : définit ce qui est bien ou mal"""
    types = [i["type"] for i in outfit]
    
    # --- RÈGLES POUR LE MARIAGE (Wedding) ---
    if occasion == "wedding":
        # Interdiction absolue du Jeans au mariage
        if "Jeans" in types: 
            return 0.0 
        
        # Priorité maximale à la robe de mariage
        if "Dress_Wedding" in types: 
            return 1.0
            
        # Les jupes élégantes sont acceptables mais moins bien qu'une robe
        if "Skirt" in types:
            return 0.6
            
        # Les robes casual sont mal vues au mariage
        if "Dress_Casual" in types:
            return 0.1
            
        return 0.2 # Autres combinaisons bof

    # --- RÈGLES POUR LE CASUAL ---
    if occasion == "casual":
        if "Dress_Wedding" in types: return 0.0 # Trop habillé pour un café
        if "Jeans" in types: return 1.0 # Le jeans est parfait en casual
        return 0.8

    # --- RÈGLES POUR LE TRAVAIL (Work) ---
    if occasion == "work":
        if "Dress_Wedding" in types: return 0.0
        if "Jeans" in types: return 0.7 # Acceptable selon le milieu
        if "Skirt" in types or "Dress_Casual" in types: return 1.0
        return 0.8

    # --- RÈGLES THERMIQUES (Sécurité) ---
    if temperature < 12 and "Coat" not in types:
        return 0.0 # Interdiction d'avoir froid !
        
    return 0.5

def outfit_to_vector(outfit, base_model):
    """Convert outfit to feature vector"""
    if not outfit:
        return None

    warmths, breaths = [], []
    for item in outfit:
        f = extract_cnn_features(item["image"], base_model)
        w, b = compute_thermal_scores(f)
        warmths.append(w)
        breaths.append(b)

    outfit_color_score = color_score(outfit)
    has_dress = int(any(i["type"] == "Dress" for i in outfit))
    has_coat = int(any(i["type"] == "Coat" for i in outfit))

    return {
        "color_score": outfit_color_score,
        "warmth": np.mean(warmths),
        "breath": np.mean(breaths),
        "has_dress": has_dress,
        "has_coat": has_coat
    }

def explain_in_words(row):
    """Generate explanation for outfit choice"""
    reasons = []
    if row["temp"] > 25:
        reasons.append("température élevée → vêtements légers")
    elif row["temp"] < 15:
        reasons.append("température fraîche → vêtements chauds")
    
    if row["has_dress"]:
        reasons.append("robe adaptée à l'occasion")
    if row["has_coat"] and row["temp"] < 15:
        reasons.append("manteau nécessaire pour le froid")
    if row["color_score"] > 0.7:
        reasons.append("couleurs harmonieuses")
    elif row["color_score"] > 0.5:
        reasons.append("combinaison de couleurs acceptable")
    
    return " | ".join(reasons) if reasons else "Tenue équilibrée"

def build_dataset_with_context(wardrobe, base_model):
    """Build training dataset"""
    rows = []
    outfits = generate_outfits(wardrobe)

    for o in outfits:
        warmths, breaths = [], []

        for item in o:
            f = extract_cnn_features(item["image"], base_model)
            w, b = compute_thermal_scores(f)
            warmths.append(w)
            breaths.append(b)

        for temp in [5, 15, 25, 35]:
            for occ in OCCASIONS:
                rows.append({
                    "temp": temp,
                    "occasion": occ,
                    "color_score": color_score(o),
                    "warmth": np.mean(warmths),
                    "breath": np.mean(breaths),
                    "has_dress": int(any(i["type"] == "Dress" for i in o)),
                    "has_coat": int(any(i["type"] == "Coat" for i in o)),
                    "label": smart_label(o, temp, occ)
                })

    return pd.DataFrame(rows)

def train_xgb(df):
    """Train XGBoost model"""
    X = df.drop("label", axis=1)
    y = df["label"]

    cat_features_model = ["occasion"]
    num_features_model = ["temp", "color_score", "warmth", "breath", "has_dress", "has_coat"]

    enc = OneHotEncoder(sparse_output=False)
    X_cat = enc.fit_transform(X[cat_features_model])

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_features_model])

    Xf = np.hstack([X_num, X_cat])
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    mse = mean_squared_error(yte, preds)
    print(f"✅ XGBoost Model MSE: {mse:.4f}") # Plus le chiffre est proche de 0, mieux c'est

    return model, enc, scaler, num_features_model, cat_features_model

# ============================================
# PYDANTIC MODELS
# ============================================

class WardrobeItem(BaseModel):
    item_id: str
    type: str
    color: str
    image_path: str
    confidence: float

class WeatherResponse(BaseModel):
    temperature: float
    city: str
    description: str
    timestamp: str
    # On rend ces champs optionnels pour éviter les erreurs 500
    feels_like: Optional[float] = None
    weather: Optional[str] = None
    humidity: Optional[int] = None
    wind_speed: Optional[float] = None

class RecommendationRequest(BaseModel):
    temperature: Optional[float] = None
    occasion: str = "casual"
    top_k: int = 3

class OutfitRecommendation(BaseModel):
    outfit_id: str
    outfit: List[Dict]
    base_score: float
    rl_adjustment: float
    final_score: float
    reasons: str

class FeedbackRequest(BaseModel):
    outfit_id: str
    outfit: List[Dict]
    temperature: float
    occasion: str
    feedback_score: int
    reason: Optional[str] = None

class StatisticsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    neutral: int
    positive_rate: float
    negative_rate: float
    neutral_rate: float

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    print("🚀 Initialisation de Outfit of the Day...")
    state.weather_service = WeatherService()
    state.feedback_store = OutfitFeedbackStore(FEEDBACK_PATH)
    
    try:
        state.cnn_general = tf.keras.models.load_model(MODEL_GENERAL_PATH)
        state.cnn_styles = tf.keras.models.load_model(MODEL_DRESS_PATH)
        state.base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
        print("✅ Modèles chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")
    
    state.is_initialized = True

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Outfit Recommender API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "weather": "/api/weather",
            "wardrobe": "/api/wardrobe",
            "recommendations": "/api/recommendations",
            "feedback": "/api/feedback",
            "statistics": "/api/statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": state.is_initialized,
        "wardrobe_items": len(state.wardrobe),
        "cnn_models_loaded": (state.cnn_general is not None or state.cnn_styles is not None),
        "xgb_model_loaded": state.xgb_model is not None,
        "feedback_entries": len(state.feedback_store.feedback_history) if state.feedback_store else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/wardrobe/upload")
async def upload(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        item_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{item_id}.jpg")
        with open(path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        
        label, conf = predict_clothing_cascading(path)
        
        # Détection couleur
        img = cv2.imread(path)
        avg_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).mean(axis=(0, 1)).astype(int)
        color_name = get_closest_color(avg_color).capitalize()

        if conf > 0.5:
            item = {"item_id": item_id, "type": label, "color": color_name, "image": path, "confidence": float(conf)}
            state.wardrobe.append(item)
            uploaded.append(item)
    return {"wardrobe": uploaded}

@app.get("/api/wardrobe")
async def get_wardrobe():
    return {"wardrobe": state.wardrobe}

@app.delete("/api/wardrobe/clear")
async def clear_wardrobe():
    """Clear all wardrobe items"""
    state.wardrobe = []
    state.xgb_model = None
    state.encoder = None
    state.scaler = None
    
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    return {"message": "Wardrobe cleared successfully"}

@app.get("/api/wardrobe/image/{item_id}")
async def get_img(item_id: str):
    path = os.path.join(UPLOAD_DIR, f"{item_id}.jpg")
    return FileResponse(path)

@app.post("/api/wardrobe/train")
async def train_model():
    """Train XGBoost model on current wardrobe"""
    
    if len(state.wardrobe) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 items to train. Current: {len(state.wardrobe)}"
        )
    
    print("\n🤖 Training XGBoost model...")
    
    try:
        # Build dataset
        df = build_dataset_with_context(state.wardrobe, state.base_model)
        print(f"   Dataset size: {len(df)} samples")
        
        # Train model
        model, enc, scaler, num_feat, cat_feat = train_xgb(df)
        
        # Save model
        state.xgb_model = model
        state.encoder = enc
        state.scaler = scaler
        state.num_features = num_feat
        state.cat_features = cat_feat
        
        # Save to disk
        with open(XGB_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(enc, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Initialize RL recommender
        state.rl_recommender = RLRecommender(state.xgb_model, state.feedback_store)
        
        print("✅ Model trained and saved successfully")
        
        return {
            "message": "Model trained successfully",
            "dataset_size": len(df),
            "wardrobe_size": len(state.wardrobe)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/weather", response_model=WeatherResponse)
async def get_weather(city: Optional[str] = None, country: Optional[str] = None):
    """Get current weather"""
    
    if not state.weather_service:
        raise HTTPException(status_code=500, detail="Weather service not initialized")
    
    weather_data = state.weather_service.get_weather(city, country)
    return weather_data


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get outfit recommendations"""
    
    if len(state.wardrobe) == 0:
        raise HTTPException(status_code=400, detail="No wardrobe items uploaded")
    
    if not state.xgb_model:
        # Try to train if we have enough items
        if len(state.wardrobe) >= 5:
            await train_model()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model not trained. Need at least 5 items. Current: {len(state.wardrobe)}"
            )
    
    # Get weather if temperature not provided
    temperature = request.temperature
    if temperature is None:
        weather = state.weather_service.get_weather()
        temperature = weather['temperature']
    
    # Initialize RL recommender if not done
    if not state.rl_recommender:
        state.rl_recommender = RLRecommender(state.xgb_model, state.feedback_store)
    
    # Get recommendations
    try:
        recommendations = state.rl_recommender.recommend_with_rl(
            state.wardrobe,
            state.encoder,
            state.scaler,
            state.num_features,
            state.cat_features,
            temperature,
            request.occasion,
            top_k=request.top_k
        )
        
        # Format recommendations
        formatted_recs = []
        for rec in recommendations:
            outfit_id = str(uuid.uuid4())
            outfit = rec['outfit']
            
            # Generate explanation
            outfit_vector = outfit_to_vector(outfit, state.base_model)
            explanation_row = {
                "temp": temperature,
                "occasion": request.occasion,
                "color_score": outfit_vector["color_score"],
                "warmth": outfit_vector["warmth"],
                "breath": outfit_vector["breath"],
                "has_dress": outfit_vector["has_dress"],
                "has_coat": outfit_vector["has_coat"]
            }
            reasons = explain_in_words(explanation_row)
            
            formatted_recs.append({
                "outfit_id": outfit_id,
                "outfit": [
                    {
                        "item_id": item["item_id"],
                        "type": item["type"],
                        "color": item["color"],
                        "image_path": item["image"]
                    }
                    for item in outfit
                ],
                "base_score": float(rec['base_score']),
                "rl_adjustment": float(rec['rl_adjustment']),
                "final_score": float(rec['final_score']),
                "reasons": reasons
            })
        
        return {
            "recommendations": formatted_recs,
            "context": {
                "temperature": temperature,
                "occasion": request.occasion,
                "total_outfits_evaluated": len(generate_outfits(state.wardrobe))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    
    if not state.feedback_store:
        raise HTTPException(status_code=500, detail="Feedback store not initialized")
    
    try:
        state.feedback_store.add_feedback(
            outfit=request.outfit,
            temperature=request.temperature,
            occasion=request.occasion,
            feedback_score=request.feedback_score,
            reason=request.reason
        )
        
        return {
            "message": "Feedback recorded successfully",
            "outfit_id": request.outfit_id,
            "feedback_score": request.feedback_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get feedback statistics"""
    
    if not state.feedback_store:
        raise HTTPException(status_code=500, detail="Feedback store not initialized")
    
    stats = state.feedback_store.get_statistics()
    return stats

@app.get("/api/preferences")
async def get_preferences():
    """Get learned preferences from feedback"""
    
    if not state.feedback_store:
        raise HTTPException(status_code=500, detail="Feedback store not initialized")
    
    preferences = state.feedback_store.get_preferred_combinations(min_score=1, min_count=1)
    
    formatted_prefs = []
    for (occasion, temp_range, outfit_types), pref in preferences.items():
        formatted_prefs.append({
            "occasion": occasion,
            "temperature_range": temp_range,
            "outfit_types": list(outfit_types),
            "count": pref['count'],
            "avg_score": round(pref['avg_score'], 2)
        })
    
    # Sort by count descending
    formatted_prefs.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        "preferences": formatted_prefs[:10],  # Top 10
        "total_feedback": len(state.feedback_store.feedback_history)
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )
# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )