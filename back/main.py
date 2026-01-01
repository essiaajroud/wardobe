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
from xgboost import XGBClassifier
import itertools

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
MODEL_PATH = "./models/fashion_cnn_model.keras"
FEEDBACK_PATH = "./data/outfit_feedback.pkl"
XGB_MODEL_PATH = "./data/xgb_model.pkl"
ENCODER_PATH = "./data/encoder.pkl"
SCALER_PATH = "./data/scaler.pkl"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Constants from notebook
CLASS_NAMES = ["Coat", "Dress", "Jeans", "Skirt", "Top"]
OCCASIONS = ["casual", "work", "party", "wedding"]

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

# ============================================
# GLOBAL STATE
# ============================================

class AppState:
    def __init__(self):
        self.wardrobe = []
        self.cnn_model = None
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
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY', 'eb09ef60723f9c25c2e6cd6da716e19e')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.geolocation_url = "http://ip-api.com/json"

    def _get_location_from_ip(self):
        """Detects city and country code based on IP address."""
        try:
            response = requests.get(self.geolocation_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data and data['status'] == 'success':
                city = data.get('city')
                country_code = data.get('countryCode')
                print(f"🌍 Detected location: {city}, {country_code} (from IP)")
                return city, country_code
            else:
                print(f"❌ Geolocation API error: {data.get('message', 'Unknown error')}. Using default.")
        except Exception as e:
            print(f"❌ Geolocation error: {e}. Using default.")
        return "Sfax", "TN"

    def get_weather(self, city=None, country_code=None):
        """Get current weather for a location. Detects location if not provided."""
        if city is None or country_code is None:
            city, country_code = self._get_location_from_ip()

        if self.api_key == 'eb09ef60723f9c25c2e6cd6da716e19e':
            print("⚠️ Using fallback weather. Get free API key at: https://openweathermap.org/api")
            return self._get_fallback_weather()

        try:
            params = {
                'q': f"{city},{country_code}",
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            weather_info = {
                'temperature': round(data['main']['temp'], 1),
                'feels_like': round(data['main']['feels_like'], 1),
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'city': city,
                'timestamp': datetime.now().isoformat()
            }

            print(f"🌤️ Real weather in {city}: {weather_info['temperature']}°C - {weather_info['description']}")
            return weather_info

        except Exception as e:
            print(f"❌ Weather API error: {e}. Using fallback.")
            return self._get_fallback_weather()

    def _get_fallback_weather(self):
        """Season-based fallback weather"""
        month = datetime.now().month

        if month in [12, 1, 2]:  # Winter
            temp = random.randint(5, 15)
            weather = "Cold"
        elif month in [3, 4, 5]:  # Spring
            temp = random.randint(15, 23)
            weather = "Mild"
        elif month in [6, 7, 8]:  # Summer
            temp = random.randint(25, 35)
            weather = "Hot"
        else:  # Autumn
            temp = random.randint(15, 22)
            weather = "Cool"

        print(f"🌡️ Fallback weather: {temp}°C - {weather}")

        return {
            'temperature': temp,
            'feels_like': temp,
            'weather': weather,
            'description': f'{weather.lower()} weather',
            'humidity': 60,
            'wind_speed': 3.5,
            'city': 'Sfax',
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
        """Get top K recommendations with RL boost"""

        all_outfits = generate_outfits(wardrobe)
        scored_outfits = []

        for outfit in all_outfits:
            outfit_feats = outfit_to_vector(outfit, state.base_model)
            if outfit_feats is None:
                continue

            # Base XGBoost score
            numerical_values = [
                temperature,
                outfit_feats["color_score"],
                outfit_feats["warmth"],
                outfit_feats["breath"],
                outfit_feats["has_dress"],
                outfit_feats["has_coat"]
            ]

            numerical_df = pd.DataFrame([numerical_values], columns=num_features)
            scaled_num = scaler.transform(numerical_df)
            encoded_cat = encoder.transform(pd.DataFrame([{"occasion": occasion}])[cat_features])
            X_predict = np.hstack([scaled_num, encoded_cat])

            base_score = self.base_model.predict_proba(X_predict)[0][1]

            # RL adjustment
            rl_adjustment = self._get_rl_adjustment(outfit, temperature, occasion)
            final_score = min(base_score * (1 + rl_adjustment), 1.0)

            scored_outfits.append({
                'outfit': outfit,
                'base_score': base_score,
                'rl_adjustment': rl_adjustment,
                'final_score': final_score
            })

        # Sort by final score
        scored_outfits.sort(key=lambda x: x['final_score'], reverse=True)

        # Exploration
        if random.random() < self.exploration_rate and len(scored_outfits) > top_k:
            print("🔍 Exploring new combination...")
            explored = random.choice(scored_outfits[top_k:min(top_k+10, len(scored_outfits))])
            scored_outfits = scored_outfits[:top_k-1] + [explored]

        return scored_outfits[:top_k]

    def _get_rl_adjustment(self, outfit, temperature, occasion):
        """Calculate RL boost based on feedback history"""
        outfit_types = tuple(sorted([item['type'] for item in outfit]))
        temp_range = self.feedback_store._get_temp_range(temperature)

        preferences = self.feedback_store.get_preferred_combinations(min_score=0, min_count=1)
        key = (occasion, temp_range, outfit_types)

        if key in preferences:
            pref = preferences[key]
            avg_score = pref['avg_score']
            count = pref['count']
            confidence = min(count / 10, 1.0)
            adjustment = avg_score * 0.15 * confidence
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

def predict_clothing(img_path, cnn_model):
    """Predict clothing type"""
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = cnn_model.predict(arr, verbose=0)
    idx = np.argmax(preds)
    conf = np.max(preds)
    return CLASS_NAMES[idx], conf

def detect_color(img_path):
    """Detect dominant color"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(pixels)
    color = kmeans.cluster_centers_[0].astype(int)
    try:
        return webcolors.rgb_to_name(tuple(color)).capitalize()
    except:
        return "Unknown"

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
    """Generate smart label for outfit"""
    types = [i["type"] for i in outfit]
    if "Dress" in types and ("Jeans" in types or "Top" in types):
        return 0
    if temperature < 10 and "Coat" not in types:
        return 0
    if temperature > 25 and "Coat" in types:
        return 0
    if occasion == "wedding" and "Dress" not in types:
        return 0
    return 1

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

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr)

    accuracy = accuracy_score(yte, model.predict(Xte))
    print(f"✅ XGBoost Model Accuracy: {accuracy:.3f}")

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
    feels_like: float
    weather: str
    description: str
    humidity: int
    wind_speed: float
    city: str
    timestamp: str

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
    """Initialize services on startup"""
    print("\n" + "="*60)
    print("🚀 AI OUTFIT RECOMMENDER API - STARTING")
    print("="*60 + "\n")
    
    # Initialize weather service
    state.weather_service = WeatherService()
    print("✅ Weather service initialized")
    
    # Initialize feedback store
    state.feedback_store = OutfitFeedbackStore(FEEDBACK_PATH)
    print(f"✅ Feedback store initialized ({len(state.feedback_store.feedback_history)} entries)")
    
    # Load CNN model if exists
    if os.path.exists(MODEL_PATH):
        try:
            state.cnn_model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ CNN model loaded")
        except Exception as e:
            print(f"⚠️ Could not load CNN model: {e}")
            print("   Model will be trained when wardrobe is uploaded")
    
    # Load MobileNetV2 base model
    state.base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    print("✅ MobileNetV2 base model loaded")
    
    # Load saved models if they exist
    if os.path.exists(XGB_MODEL_PATH):
        try:
            with open(XGB_MODEL_PATH, 'rb') as f:
                state.xgb_model = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f:
                state.encoder = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                state.scaler = pickle.load(f)
            print("✅ XGBoost model, encoder, and scaler loaded")
        except Exception as e:
            print(f"⚠️ Could not load saved models: {e}")
    
    state.is_initialized = True
    print("\n✅ API READY!\n")

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
        "cnn_model_loaded": state.cnn_model is not None,
        "xgb_model_loaded": state.xgb_model is not None,
        "feedback_entries": len(state.feedback_store.feedback_history) if state.feedback_store else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/wardrobe/upload")
async def upload_wardrobe(files: List[UploadFile] = File(...)):
    """Upload wardrobe images"""
    
    if not state.cnn_model:
        raise HTTPException(
            status_code=400, 
            detail="CNN model not loaded. Please ensure fashion_cnn_model.keras is in ./models/"
        )
    
    uploaded_items = []
    
    for file in files:
        # Generate unique ID
        item_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            continue
            
        file_path = os.path.join(UPLOAD_DIR, f"{item_id}{file_ext}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Predict clothing type and detect color
            clothing_type, confidence = predict_clothing(file_path, state.cnn_model)
            color = detect_color(file_path)
            
            if confidence > 0.6:  # Only add if confidence is high enough
                item = {
                    "item_id": item_id,
                    "type": clothing_type,
                    "color": color,
                    "image": file_path,
                    "confidence": float(confidence),
                    "filename": file.filename
                }
                
                uploaded_items.append(item)
                state.wardrobe.append(item)
            else:
                os.remove(file_path)  # Remove low confidence predictions
                
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Train model if we have enough items
    if len(state.wardrobe) >= 5 and not state.xgb_model:
        await train_model()
    
    return {
        "message": f"Successfully uploaded {len(uploaded_items)} items",
        "wardrobe": uploaded_items,
        "total_items": len(state.wardrobe)
    }

@app.get("/api/wardrobe")
async def get_wardrobe():
    """Get current wardrobe"""
    return {
        "wardrobe": state.wardrobe,
        "total_items": len(state.wardrobe),
        "by_type": {
            clothing_type: len([w for w in state.wardrobe if w["type"] == clothing_type])
            for clothing_type in CLASS_NAMES
        }
    }

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
async def get_wardrobe_image(item_id: str):
    """Get wardrobe item image"""
    item = next((w for w in state.wardrobe if w["item_id"] == item_id), None)
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    if not os.path.exists(item["image"]):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(item["image"])

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
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled error: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════╗
    ║   AI Outfit Recommender API Server       ║
    ║   Version 2.0.0 - Adjusted to Notebook   ║
    ╚══════════════════════════════════════════╝
    
    🚀 Starting server...
    📡 API Documentation: http://localhost:8000/docs
    🌐 Health Check: http://localhost:8000/health
    
    📝 Required setup:
    1. Place fashion_cnn_model.keras in ./models/
    2. Upload wardrobe images via /api/wardrobe/upload
    3. Train model via /api/wardrobe/train
    4. Get recommendations via /api/recommendations
    
    """)
    
    uvicorn.run(
        "outfit_backend_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )