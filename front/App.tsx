import React, { useState, useEffect, useRef } from 'react';
import { ClothingItem, Occasion, Weather, OutfitRecommendation, WeatherData } from './types';

// --- Configuration API Backend ---
const API_BASE_URL = 'http://localhost:8000';

// --- UI Components ---

const Button: React.FC<{
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  className?: string;
  disabled?: boolean;
  children: React.ReactNode;
}> = ({ onClick, variant = 'primary', className = '', disabled, children }) => {
  const base = "px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 flex items-center justify-center gap-2 active:scale-95 disabled:opacity-50 disabled:active:scale-100";
  const variants = {
    primary: "bg-black text-white hover:bg-neutral-800 shadow-lg shadow-black/10",
    secondary: "bg-white text-black hover:bg-neutral-50 shadow-sm",
    outline: "border border-neutral-200 text-neutral-600 hover:border-black hover:text-black",
    ghost: "text-neutral-400 hover:text-black hover:bg-neutral-50"
  };
  return (
    <button onClick={onClick} disabled={disabled} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
};

const TabButton: React.FC<{ active: boolean; onClick: () => void; label: string; icon: string }> = ({ active, onClick, label, icon }) => (
  <button 
    onClick={onClick}
    className={`flex items-center gap-2 px-6 py-3 rounded-full text-xs font-bold uppercase tracking-widest transition-all ${active ? 'bg-black text-white shadow-md' : 'text-neutral-400 hover:text-neutral-600'}`}
  >
    <i className={`fas ${icon}`}></i>
    {label}
  </button>
);

const SelectionChip: React.FC<{ active: boolean; onClick: () => void; label: string; icon: string }> = ({ active, onClick, label, icon }) => (
  <button 
    onClick={onClick}
    className={`flex flex-col items-center justify-center gap-2 p-4 rounded-2xl border transition-all duration-300 min-w-[80px] sm:min-w-[100px] ${
      active 
        ? 'bg-black border-black text-white shadow-xl scale-105 z-10' 
        : 'bg-white border-neutral-100 text-neutral-500 hover:border-neutral-300 hover:bg-neutral-50'
    }`}
  >
    <i className={`fas ${icon} text-lg ${active ? 'text-white' : 'text-neutral-300'}`}></i>
    <span className="text-[10px] font-bold uppercase tracking-tight">{label}</span>
  </button>
);

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'closet' | 'stylist'>('closet');
  const [wardrobe, setWardrobe] = useState<ClothingItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [occasion, setOccasion] = useState<Occasion>(Occasion.CASUAL);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const occasionIcons: Record<Occasion, string> = {
    [Occasion.CASUAL]: 'fa-couch',
    [Occasion.WORK]: 'fa-briefcase',
    [Occasion.PARTY]: 'fa-glass-cheers',
    [Occasion.WEDDING]: 'fa-ring',
    
  };

  // --- API Handlers ---

  const fetchWeather = async () => {
  try {
    // On ne passe plus de ville en dur, le backend utilisera l'IP
    const res = await fetch(`${API_BASE_URL}/api/weather`); 
    if (!res.ok) throw new Error("Weather fetch failed");
    const data = await res.json();
    setWeatherData(data);
  } catch (e) {
    console.error("Weather error:", e);
  }
};

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/statistics`);
      if (!res.ok) throw new Error("Stats fetch failed");
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error("Stats error:", e);
    }
  };

  const loadWardrobe = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/wardrobe`);
      const data = await res.json();
      const mapped = data.wardrobe.map((item: any) => ({
        id: item.item_id,
        name: `${item.type}`,
        category: item.type,
        color: item.color,
        imageData: `${API_BASE_URL}/api/wardrobe/image/${item.item_id}`,
        description: `Confidence: ${(item.confidence * 100).toFixed(0)}%`,
        style: 'AI Analyzed',
        seasons: ['All']
      }));
      setWardrobe(mapped);
    } catch (e) {
      console.error("Load Wardrobe Error:", e);
    }
  };

   useEffect(() => {
    fetchWeather();
    const interval = setInterval(fetchWeather, 600000); 
    return () => clearInterval(interval);

  }, []);

  useEffect(() => {
    // fetchWeather();
    // const interval = setInterval(fetchWeather, 600000); 
    // return () => clearInterval(interval);
    fetchStats();
    loadWardrobe();
  }, []);


const uploadToBackend = async (blob: Blob) => {
  setAnalyzing(true);
  const formData = new FormData();
  formData.append('files', blob, `upload_${Date.now()}.jpg`);

  try {
    const res = await fetch(`${API_BASE_URL}/api/wardrobe/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error("Upload failed");
    
    const data = await res.json();

    // ✅ FIX : Vérifier si la liste n'est pas vide avant d'accéder à l'index [0]
    if (data.wardrobe && data.wardrobe.length > 0) {
      console.log(`Vêtement identifié : ${data.wardrobe[0].type}`);
      loadWardrobe(); // Recharger la liste
    } else {
      console.log("L'IA n'est pas sûre d'elle (confiance < 60%). Réessayez avec une meilleure lumière.");
    }

  } catch (err) {
    console.error("Upload Error:", err);
    alert("Erreur lors de la connexion au serveur.");
  } finally {
    setAnalyzing(false);
  }
};

  const handleRecommend = async () => {
    if (!weatherData) {
      console.log("Météo en cours de chargement...");
      return;
    }
    if (wardrobe.length === 0) {
      console.log("Votre dressing est vide !");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: weatherData.temperature,
          occasion: occasion,
          top_k: 3
        }),
      });
      if (!res.ok) throw new Error("Recommendation failed");
      
      const data = await res.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      console.error("Recommendation Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (rec: any, score: number) => {
  try {
    const res = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        outfit_id: rec.outfit_id,
        outfit: rec.outfit,
        temperature: weatherData?.temperature || 20,
        occasion: occasion,
        feedback_score: score
      }),
    });
    if (res.ok) {
      fetchStats();
      console.log(score > 0 
        ? "Génial ! Outfit of the Day apprend de vos goûts." 
        : "C'est noté, Outfit of the Day ajuste son modèle."
      );
    }
  } catch (e) {
    console.error("Feedback error:", e);
    console.error("Erreur lors de la connexion au serveur.");}
};

  const trainModel = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/wardrobe/train`, { method: 'POST' });
      const data = await res.json();
      console.log("Modèle XGBoost ré-entraîné avec succès !", data);
    } catch (e) {
      console.log("Besoin d'au moins 5 articles pour l'entraînement.");
    } finally {
      setLoading(false);
    }
  };

  // --- Interface Handlers ---

  // Fix: Explicitly type 'file' as 'File' in the forEach callback to resolve 'unknown' type inference error on line 217.
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      Array.from(files).forEach((file: File) => {
        uploadToBackend(file);
      });
    }
  };

  const startCamera = async () => {
    setIsCameraOpen(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      setIsCameraOpen(false);
      console.error("Accès caméra refusé.", err);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      ctx?.drawImage(videoRef.current, 0, 0);
      canvasRef.current.toBlob((blob) => {
        if (blob) uploadToBackend(blob);
      }, 'image/jpeg');
      stopCamera();
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
    }
    setIsCameraOpen(false);
  };

  return (
    <div className="min-h-screen bg-white selection:bg-black selection:text-white">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-40 bg-white/90 backdrop-blur-2xl border-b border-neutral-100">
  <div className="max-w-7xl mx-auto px-6 h-24 flex items-center justify-between">
    <div className="flex items-center gap-3">
      {/* Nouveau symbole : Vêtement (fa-tshirt) */}
      <div className="w-10 h-10 bg-black rounded-xl flex items-center justify-center shadow-lg">
        <i className="fas fa-tshirt text-white text-sm"></i>
      </div>
      {/* Nouveau nom : Outfit of the Day */}
      <span className="text-2xl font-serif italic tracking-tighter">Outfit of the Day</span>
    </div>
    
    <nav className="flex bg-neutral-100 p-1.5 rounded-full border border-neutral-200/50">
      <TabButton active={activeTab === 'closet'} onClick={() => setActiveTab('closet')} label="Dressing" icon="fa-th-large" />
      <TabButton active={activeTab === 'stylist'} onClick={() => setActiveTab('stylist')} label="Styliste" icon="fa-wand-magic-sparkles" />
    </nav>

    <div className="hidden md:flex items-center gap-6">
       {stats && (
         <div className="text-right">
           <div className="text-[9px] font-black text-neutral-300 uppercase tracking-widest">Précision Modèle</div>
           <div className="text-sm font-bold text-black">{stats.positive_rate}%</div>
         </div>
       )}
       {/* LE CERCLE AVATAR A ÉTÉ SUPPRIMÉ ICI */}
    </div>
  </div>
</header>

      <main className="pt-36 pb-24 max-w-7xl mx-auto px-6">
        {activeTab === 'closet' && (
          <section className="animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-16">
              <div>
                <h1 className="text-6xl font-serif mb-4 tracking-tight">Dressing Numérique</h1>
                <p className="text-neutral-400 font-bold tracking-[0.2em] uppercase text-[10px] flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                  {wardrobe.length} articles classifiés par MobileNetV2
                </p>
              </div>
              <div className="flex gap-4 w-full md:w-auto">
                <input type="file" multiple className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
                <Button variant="outline" onClick={startCamera} className="flex-1 md:flex-none py-4">
                  <i className="fas fa-camera"></i> Scan direct
                </Button>
                <Button variant="outline" onClick={trainModel} className="flex-1 md:flex-none py-4">
                  <i className="fas fa-sync"></i> Entraîner
                </Button>
                <Button variant="primary" onClick={() => fileInputRef.current?.click()} className="flex-1 md:flex-none py-4" disabled={analyzing}>
                  {analyzing ? <i className="fas fa-circle-notch fa-spin"></i> : <i className="fas fa-upload"></i>}
                  {analyzing ? 'Analyse...' : 'Ajouter'}
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-8">
              {wardrobe.map(item => (
                <div key={item.id} className="group relative bg-white rounded-[2rem] overflow-hidden shadow-sm hover:shadow-2xl hover:-translate-y-2 transition-all duration-700 border border-neutral-100">
                  <div className="aspect-[3/4] overflow-hidden bg-neutral-50 flex items-center justify-center">
                    <img src={item.imageData} alt={item.name} className="w-full h-full object-cover" />
                  </div>
                  <div className="p-5 bg-white">
                    <div className="text-[10px] font-black uppercase tracking-widest text-neutral-300 mb-1.5">{item.category}</div>
                    <h4 className="text-sm font-bold text-neutral-900 truncate tracking-tight">{item.name}</h4>
                    <div className="text-[9px] text-neutral-400 font-medium mt-1 lowercase italic">{item.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'stylist' && (
          <section className="animate-in fade-in slide-in-from-bottom-4 duration-700 grid lg:grid-cols-12 gap-16">
            <div className="lg:col-span-5 space-y-16">
              <div>
                <h2 className="text-5xl font-serif mb-4 tracking-tight leading-tight">Architecte<br/>de Look.</h2>
                <div className="flex items-center gap-4 bg-neutral-50 p-4 rounded-2xl border border-neutral-100 w-fit">
                  {weatherData ? (
                    <>
                      <div className="text-3xl text-black font-bold">{weatherData.temperature}°C</div>
                      <div className="text-xs font-bold text-neutral-400 uppercase tracking-widest">
                        {weatherData.description} • {weatherData.city}
                      </div>
                    </>
                  ) : (
                    <div className="text-xs font-bold text-neutral-400 animate-pulse">Connexion WeatherService...</div>
                  )}
                </div>
              </div>

              <div className="space-y-12">
                <div className="bg-neutral-50/50 p-8 rounded-[3rem] border border-neutral-100">
                  <label className="text-[10px] font-black uppercase tracking-[0.3em] text-neutral-400 block mb-6">Occasion & Intention</label>
                  <div className="grid grid-cols-3 gap-3">
                    {Object.values(Occasion).map(o => (
                      <SelectionChip 
                        key={o} 
                        onClick={() => setOccasion(o)} 
                        active={occasion === o} 
                        label={o} 
                        icon={occasionIcons[o]} 
                      />
                    ))}
                  </div>
                </div>

                <Button variant="primary" onClick={handleRecommend} className="w-full py-6 text-base shadow-2xl shadow-black/20" disabled={loading || !weatherData}>
                  {loading ? <i className="fas fa-circle-notch fa-spin"></i> : <i className="fas fa-sparkles mr-2"></i>}
                  {loading ? 'Consultation XGBoost...' : 'Générer Tenue'}
                </Button>
              </div>
            </div>

            <div className="lg:col-span-7">
              <div className={`min-h-[70vh] rounded-[4rem] bg-neutral-50/30 border-2 border-neutral-50 p-8 md:p-16 flex flex-col ${recommendations.length > 0 ? 'justify-start' : 'justify-center items-center text-center'}`}>
                {recommendations.length === 0 ? (
                  <div className="max-w-xs text-center">
                    {/* Icône de robe pour l'attente */}
                    <div className="w-24 h-24 mb-10 bg-white rounded-[2.5rem] flex items-center justify-center mx-auto shadow-2xl text-neutral-200">
                      <i className="fas fa-vest text-4xl"></i>
                    </div>
                    <h3 className="text-2xl font-medium mb-4">En attente</h3>
                    <p className="text-neutral-400 text-sm leading-relaxed">
                      Choisissez une occasion. <strong>Outfit of the Day</strong> consultera son modèle de Reinforcement Learning pour trouver la meilleure tenue.
                    </p>
                  </div>
                ) : (
                  <div className="animate-in fade-in duration-1000 space-y-16 w-full">
                    {recommendations.map((rec, idx) => (
                      <div key={rec.outfit_id} className="space-y-10 border-b border-neutral-100 pb-16 last:border-0">
                        <div className="flex flex-wrap justify-center gap-6">
                          {rec.outfit.map((piece: any, i: number) => (
                            <div key={i} className="w-40 md:w-56 group">
                              <div className="aspect-[3/4] rounded-[2rem] bg-white overflow-hidden border-4 border-white shadow-xl mb-4 flex items-center justify-center">
                                <img 
                                  src={`${API_BASE_URL}/api/wardrobe/image/${piece.item_id}`} 
                                  className="w-full h-full object-cover" 
                                  onError={(e) => (e.currentTarget.src = 'https://via.placeholder.com/200x300?text=Vêtement')}
                                />
                              </div>
                              <span className="text-[10px] font-black uppercase text-neutral-300 tracking-[0.2em]">{piece.type}</span>
                              {/* Color suppressed from UI per request */}
                            </div>
                          ))}
                        </div>

                        <div className="max-w-xl mx-auto bg-white p-8 md:p-12 rounded-[3.5rem] shadow-xl border border-neutral-100 relative">
                           <div className="absolute -top-6 -left-6 w-14 h-14 bg-black rounded-full flex items-center justify-center shadow-lg">
                              <i className="fas fa-brain text-white text-lg"></i>
                           </div>
                          <h4 className="text-2xl font-serif italic mb-6">Logique du modèle</h4>
                          <p className="text-neutral-600 leading-[1.8] mb-8 italic">"{rec.reasons}"</p>
                          
                          <div className="pt-8 border-t border-neutral-100 flex justify-between items-center">
                            <div>
                               <div className="text-[9px] font-black text-neutral-300 uppercase tracking-widest mb-1">Score de Confiance</div>
                               <div className="text-xl font-bold">{rec.final_score >= 0.95 ? "99" : (rec.final_score * 100).toFixed(0)}%</div>
                            </div>
                            <div className="flex gap-3">
                               <button 
                                 onClick={() => handleFeedback(rec, 1)}
                                 className="w-12 h-12 rounded-full border border-neutral-100 flex items-center justify-center hover:bg-green-50 hover:text-green-500 hover:border-green-200 transition-all"
                               >
                                 <i className="fas fa-thumbs-up"></i>
                               </button>
                               <button 
                                 onClick={() => handleFeedback(rec, -1)}
                                 className="w-12 h-12 rounded-full border border-neutral-100 flex items-center justify-center hover:bg-red-50 hover:text-red-500 hover:border-red-200 transition-all"
                               >
                                 <i className="fas fa-thumbs-down"></i>
                               </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Camera Modal */}
      {isCameraOpen && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          <video ref={videoRef} autoPlay playsInline className="h-full w-full object-cover opacity-60" />
          <canvas ref={canvasRef} className="hidden" />
          <div className="absolute inset-0 border-[4rem] border-black/40 pointer-events-none flex items-center justify-center">
             <div className="w-72 h-96 border border-white/30 rounded-[3rem]" />
          </div>
          <div className="absolute bottom-16 left-0 right-0 flex items-center justify-center gap-16">
            <button onClick={stopCamera} className="w-16 h-16 rounded-full border border-white/20 text-white flex items-center justify-center">
              <i className="fas fa-times text-xl"></i>
            </button>
            <button onClick={capturePhoto} className="w-24 h-24 rounded-full bg-white p-1.5 shadow-2xl">
              <div className="w-full h-full rounded-full border-4 border-black/5 flex items-center justify-center">
                 <div className="w-2 h-2 bg-black rounded-full"></div>
              </div>
            </button>
            <div className="w-16 h-16" />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;