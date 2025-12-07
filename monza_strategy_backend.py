import os
import pickle
import joblib
import numpy as np
import pandas as pd

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: TensorFlow not found. Tire degradation classifier will be disabled.")
    TENSORFLOW_AVAILABLE = False
    tf = None

import fastf1
from fastf1 import plotting
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

# Enable FastF1 cache (auto-create directory if needed)
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"📁 Created cache directory: {cache_dir}")
fastf1.Cache.enable_cache(cache_dir)

# ----------------------------------------------------------------------
# 1. FastF1 Data Fetcher for Monza
# ----------------------------------------------------------------------

class MonzaDataFetcher:
    """
    Fetches real race data from Monza using FastF1.
    """
    
    def __init__(self, year: int = 2024, session_type: str = 'R'):
        """
        Args:
            year: Race year
            session_type: 'R' for Race, 'Q' for Qualifying, 'FP1', 'FP2', 'FP3' for practice
        """
        self.year = year
        self.session_type = session_type
        print(f"\n🏁 Loading Monza {year} {session_type} session data...")
        
    def load_session(self):
        """Load the Monza session"""
        try:
            session = fastf1.get_session(self.year, 'Monza', self.session_type)
            session.load()
            return session
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return None
    
    def get_driver_lap_data(self, session, driver: str, lap_number: Optional[int] = None) -> pd.DataFrame:
        """
        Get lap data for a specific driver.
        
        Args:
            session: FastF1 session object
            driver: Driver code (e.g., 'VER', 'HAM', 'LEC')
            lap_number: Specific lap number (optional)
        """
        try:
            driver_laps = session.laps.pick_driver(driver)
            
            if lap_number:
                return driver_laps[driver_laps['LapNumber'] == lap_number]
            
            return driver_laps
        except Exception as e:
            print(f"❌ Error fetching driver data: {e}")
            return pd.DataFrame()
    
    def extract_lap_features(self, lap_data: pd.Series, previous_laps: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from a lap for ML prediction.
        MUST match the training features exactly - using REAL telemetry!
        """
        features = {
            # Basic info
            'LapNumber': int(lap_data['LapNumber']),
            'LapTime': lap_data['LapTime'].total_seconds() if pd.notna(lap_data['LapTime']) else None,
            'Compound': lap_data['Compound'],
            'TyreLife': int(lap_data['TyreLife']) if pd.notna(lap_data['TyreLife']) else 0,
            'Stint': int(lap_data['Stint']),
            'IsPersonalBest': bool(lap_data['IsPersonalBest']),
            'PitOutTime': lap_data['PitOutTime'] if pd.notna(lap_data['PitOutTime']) else None,
            'PitInTime': lap_data['PitInTime'] if pd.notna(lap_data['PitInTime']) else None,
            'Driver': str(lap_data['Driver']) if 'Driver' in lap_data else 'Unknown',
            'DriverNumber': str(lap_data['DriverNumber']) if 'DriverNumber' in lap_data else '0',
        }
        
        # Position
        if 'Position' in lap_data.index and pd.notna(lap_data['Position']):
            features['Position'] = int(lap_data['Position'])
        else:
            features['Position'] = 10  # Default mid-field
        
        # Extract REAL telemetry from previous laps
        print("   → Extracting real telemetry data from previous laps...")
        telemetry_data = []
        speed_data = []
        throttle_data = []
        brake_data = []
        rpm_data = []
        
        # Get telemetry from last 5 laps (for robust averages)
        laps_to_check = previous_laps.tail(5) if len(previous_laps) >= 5 else previous_laps
        
        for idx, prev_lap in laps_to_check.iterrows():
            try:
                tel = prev_lap.get_telemetry()
                if tel is not None and len(tel) > 10:  # Valid telemetry
                    if 'Speed' in tel.columns:
                        speed_data.extend(tel['Speed'].dropna().tolist())
                    if 'Throttle' in tel.columns:
                        throttle_data.extend(tel['Throttle'].dropna().tolist())
                    if 'Brake' in tel.columns:
                        brake_data.extend(tel['Brake'].dropna().tolist())
                    if 'RPM' in tel.columns:
                        rpm_data.extend(tel['RPM'].dropna().tolist())
            except Exception as e:
                # Skip this lap if telemetry fails
                continue
        
        # Calculate REAL statistics from telemetry
        if len(speed_data) > 0:
            features['MaxSpeed'] = max(speed_data)
            features['AvgSpeed'] = np.mean(speed_data)
            features['SpeedStd'] = np.std(speed_data)
            print(f"      ✓ Speed data: Max={features['MaxSpeed']:.1f}, Avg={features['AvgSpeed']:.1f}")
        else:
            # Fallback to defaults only if telemetry completely fails
            features['MaxSpeed'] = 340.0
            features['AvgSpeed'] = 280.0
            features['SpeedStd'] = 15.0
            print(f"      ⚠ Using default speed values (no telemetry available)")
        
        if len(throttle_data) > 0:
            features['ThrottleMean'] = np.mean(throttle_data)
            print(f"      ✓ Throttle: {features['ThrottleMean']:.1f}%")
        else:
            features['ThrottleMean'] = 85.0
            print(f"      ⚠ Using default throttle value")
        
        if len(brake_data) > 0:
            features['BrakeMean'] = np.mean(brake_data)
            print(f"      ✓ Brake: {features['BrakeMean']:.1f}%")
        else:
            features['BrakeMean'] = 15.0
        
        if len(rpm_data) > 0:
            features['RpmMean'] = np.mean(rpm_data)
        else:
            features['RpmMean'] = 11500.0
        
        # Calculate additional metrics from previous laps
        if len(previous_laps) > 0:
            valid_laps = previous_laps.dropna(subset=['LapTime'])
            
            if len(valid_laps) > 0:
                # Lap time stats
                lap_times = [t.total_seconds() for t in valid_laps['LapTime']]
                features['AvgPreviousLapTime'] = np.mean(lap_times)
                features['StdPreviousLapTime'] = np.std(lap_times) if len(lap_times) > 1 else 0
                
                # Rolling features (last 3 laps)
                recent_times = lap_times[-3:] if len(lap_times) >= 3 else lap_times
                features['RollingAvg_3'] = np.mean(recent_times)
                features['RollingStd_3'] = np.std(recent_times) if len(recent_times) > 1 else 0
                
                # Lag features
                if len(lap_times) > 0:
                    features['PrevLapTime'] = lap_times[-1]
                    if len(lap_times) > 1:
                        features['LapTimeDelta'] = lap_times[-1] - lap_times[-2]
                        
                # Calculate sector times from current lap
                if hasattr(lap_data, 'Sector1Time') and pd.notna(lap_data['Sector1Time']):
                    features['Sector1'] = lap_data['Sector1Time'].total_seconds()
                if hasattr(lap_data, 'Sector2Time') and pd.notna(lap_data['Sector2Time']):
                    features['Sector2'] = lap_data['Sector2Time'].total_seconds()
                if hasattr(lap_data, 'Sector3Time') and pd.notna(lap_data['Sector3Time']):
                    features['Sector3'] = lap_data['Sector3Time'].total_seconds()
                
                # Sector variance
                if all(k in features for k in ['Sector1', 'Sector2', 'Sector3']):
                    features['SectorVar'] = np.std([features['Sector1'], features['Sector2'], features['Sector3']])
                else:
                    features['SectorVar'] = 1.5
            
            # Speed metrics from Speed traps (if available)
            for speed_col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                if speed_col in previous_laps.columns:
                    speed_vals = previous_laps[speed_col].dropna()
                    if len(speed_vals) > 0:
                        features[f'Avg{speed_col}'] = speed_vals.mean()
                        if speed_col == 'SpeedST':  # Speed trap - most important
                            recent_speeds = previous_laps[speed_col].tail(3).dropna()
                            if len(recent_speeds) > 0:
                                features['RollingAvgSpeed_3'] = recent_speeds.mean()
                                features['PrevAvgSpeed'] = previous_laps[speed_col].iloc[-1] if len(previous_laps) > 0 else 0
                                if len(recent_speeds) > 1:
                                    features['SpeedDelta'] = recent_speeds.iloc[-1] - recent_speeds.iloc[:-1].mean()
        
        print(f"   → Feature extraction complete: {len(features)} features extracted")
        return features

# ----------------------------------------------------------------------
# 2. Enhanced Scenario Detection
# ----------------------------------------------------------------------

def detect_lap_scenario(
    lap_data: Dict[str, Any],
    predicted_pace: float,
    previous_laps: pd.DataFrame
) -> Tuple[str, Dict[str, Any]]:
    """
    Detects why a lap time is significantly slow or abnormal at Monza.
    Returns scenario and detailed analysis.
    """
    
    analysis = {
        'lap_time': lap_data.get('LapTime'),
        'predicted_pace': predicted_pace,
        'tire_age': lap_data.get('TyreLife', 0),
        'compound': lap_data.get('Compound', 'UNKNOWN'),
    }
    
    lap_time = lap_data.get('LapTime')
    if lap_time is None:
        return "NO_LAP_TIME", analysis
    
    # Check for pit lap
    is_pit_out = lap_data.get('PitOutTime') is not None
    is_pit_in = lap_data.get('PitInTime') is not None
    
    if is_pit_out or is_pit_in:
        analysis['is_pit_lap'] = True
        if lap_time > predicted_pace * 1.15:
            return "PIT_OUTLAP_OR_INLAP", analysis
    
    pct_diff = (lap_time - predicted_pace) / predicted_pace
    analysis['pace_delta_pct'] = pct_diff * 100
    
    # Calculate speed delta if available
    speed_delta = 0
    if len(previous_laps) > 1 and 'SpeedST' in previous_laps.columns:
        current_speed = previous_laps.iloc[-1]['SpeedST'] if pd.notna(previous_laps.iloc[-1]['SpeedST']) else 0
        avg_speed = previous_laps['SpeedST'].mean()
        speed_delta = current_speed - avg_speed
        analysis['speed_delta'] = speed_delta
    
    # Tire age analysis
    tire_age = lap_data.get('TyreLife', 0)
    
    # Check for inlap (preparing to pit)
    if tire_age > 15 and pct_diff > 0.12 and speed_delta < -2:
        return "INLAP", analysis
    
    # Check for damage/puncture
    if pct_diff > 0.08 and speed_delta < -10:
        return "PUNCTURE_OR_DAMAGE", analysis
    
    # Check for safety car (consistent slow pace)
    if len(previous_laps) > 3:
        recent_times = [t.total_seconds() for t in previous_laps.tail(4)['LapTime'] if pd.notna(t)]
        if len(recent_times) >= 3:
            lap_std = np.std(recent_times)
            avg_recent = np.mean(recent_times)
            if lap_std < 2 and avg_recent > predicted_pace * 1.20:
                analysis['lap_std'] = lap_std
                return "SAFETY_CAR", analysis
    
    # Check for traffic
    if 0.05 < pct_diff < 0.15:
        return "TRAFFIC", analysis
    
    # Normal pace
    if abs(pct_diff) < 0.03:
        return "NORMAL_PACE", analysis
    
    # Exceptional pace (faster than expected)
    if pct_diff < -0.02:
        return "STRONG_PACE", analysis
    
    return "UNKNOWN_ANOMALY", analysis

# ----------------------------------------------------------------------
# 3. ML Lap Time Predictor (with your existing model)
# ----------------------------------------------------------------------

class LapTimePredictor:
    """
    Uses saved Gradient Boosting model for Monza pace prediction.
    """

    def __init__(self, model_dir: str = "f1_monza_production_model"):
        print("\n🤖 Loading ML pace prediction model...")
        try:
            self.model = joblib.load(os.path.join(model_dir, "gradient_boosting_model.pkl"))
            self.feature_scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
            self.target_scaler = joblib.load(os.path.join(model_dir, "target_scaler.pkl"))

            with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
                self.feature_names = pickle.load(f)

            print("✅ ML model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            raise

    def build_feature_vector(self, raw: Dict[str, Any]) -> np.ndarray:
        """Build feature vector matching training schema."""
        s = pd.Series(0.0, index=self.feature_names)
        
        # Fill in values that exist in raw
        for k, v in raw.items():
            if k in s.index:
                try:
                    s[k] = float(v)
                except:
                    pass
        
        # Handle missing driver and compound one-hot encodings
        # Set all other driver columns to 0 (they're not in raw dict)
        for col in self.feature_names:
            if col.startswith('Driver_') and col not in raw:
                s[col] = 0.0
            if col.startswith('Compound_') and col not in raw:
                s[col] = 0.0
        
        return s.values.reshape(1, -1)

    def predict_pace(self, raw_features: Dict[str, Any]) -> float:
        """Predict lap time."""
        X = self.build_feature_vector(raw_features)
        
        # Fill any remaining NaN values with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        Xs = self.feature_scaler.transform(X)
        y_scaled = self.model.predict(Xs).reshape(1, -1)
        y = self.target_scaler.inverse_transform(y_scaled)
        return float(y.flatten()[0])

# ----------------------------------------------------------------------
# 4. DL Tire Degradation Classifier (with your existing model)
# ----------------------------------------------------------------------

class TireDegradationClassifier:
    """
    CNN + LSTM model for tire condition classification.
    """

    LABELS = ["Fresh", "Optimal", "Worn", "Critical"]

    def __init__(self, model_file: str = "tire_degradation_model_final.h5"):
        print("\n🧠 Loading DL tire degradation classifier...")
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available - using fallback tire analysis")
            self.model = None
            return
            
        try:
            self.model = tf.keras.models.load_model(model_file)
            print("✅ DL classifier loaded successfully.")
        except Exception as e:
            print(f"⚠️  Could not load tire model: {e}")
            print("   Using fallback tire analysis")
            self.model = None

    def predict_state(self, last_10_laps: List[float]) -> Dict[str, Any]:
        """Predict tire condition from 10-lap sequence."""
        
        # Fallback method if TensorFlow is not available
        if self.model is None:
            return self._fallback_prediction(last_10_laps)
        
        # Ensure we have exactly 10 laps
        if len(last_10_laps) < 10:
            last_10_laps = [last_10_laps[0]] * (10 - len(last_10_laps)) + last_10_laps
        elif len(last_10_laps) > 10:
            last_10_laps = last_10_laps[-10:]
        
        seq = np.array(last_10_laps, dtype=np.float32).reshape(1, 10, 1)
        preds = self.model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        probs = {self.LABELS[i]: float(p) for i, p in enumerate(preds)}

        return {
            "condition": self.LABELS[idx],
            "confidence": conf,
            "probabilities": probs,
        }
    
    def _fallback_prediction(self, last_10_laps: List[float]) -> Dict[str, Any]:
        """
        Rule-based tire condition prediction for 2024 Monza.
        Based on REAL data: Very low degradation due to resurfaced track.
        """
        if len(last_10_laps) < 2:
            return {
                "condition": "Fresh",
                "confidence": 0.95,
                "probabilities": {"Fresh": 0.95, "Optimal": 0.03, "Worn": 0.01, "Critical": 0.01}
            }
        
        # Calculate degradation trend
        recent_avg = np.mean(last_10_laps[-3:]) if len(last_10_laps) >= 3 else last_10_laps[-1]
        early_avg = np.mean(last_10_laps[:3]) if len(last_10_laps) >= 3 else last_10_laps[0]
        degradation = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        
        # 2024 Monza had VERY LOW degradation - adjust thresholds
        # Fresh: 0-5 laps, Optimal: 6-20 laps, Worn: 21-35 laps, Critical: 36+ laps
        if degradation < 0.005:  # Very minimal degradation
            condition = "Fresh"
            probs = {"Fresh": 0.85, "Optimal": 0.12, "Worn": 0.02, "Critical": 0.01}
        elif degradation < 0.015:  # Still performing well
            condition = "Optimal"
            probs = {"Fresh": 0.05, "Optimal": 0.80, "Worn": 0.12, "Critical": 0.03}
        elif degradation < 0.035:  # Starting to wear
            condition = "Worn"
            probs = {"Fresh": 0.02, "Optimal": 0.10, "Worn": 0.75, "Critical": 0.13}
        else:  # Significant wear
            condition = "Critical"
            probs = {"Fresh": 0.01, "Optimal": 0.04, "Worn": 0.20, "Critical": 0.75}
        
        return {
            "condition": condition,
            "confidence": probs[condition],
            "probabilities": probs
        }

# ----------------------------------------------------------------------
# 5. Strategy Recommendation Engine
# ----------------------------------------------------------------------

def recommend_compound(current_compound: Optional[str], scenario: str, tire_condition: str, tire_age: int, laps_remaining: int) -> str:
    """
    Recommend next tire compound based on 2024 Monza characteristics.
    
    Real 2024 Data:
    - SOFT (C5): ~20 laps optimal, performance drops after
    - MEDIUM (C4): ~30-50 laps, good balance
    - HARD (C5): Can do full race (53+ laps), but slower
    - Very low degradation due to resurfaced track
    """
    compounds = ["SOFT", "MEDIUM", "HARD"]
    current = current_compound.upper() if current_compound else "MEDIUM"
    
    # Emergency situations - go to most durable
    if scenario in ["PUNCTURE_OR_DAMAGE", "INLAP"]:
        return "HARD"
    
    # Safety car - opportunity for undercut
    if scenario == "SAFETY_CAR":
        if laps_remaining > 30:
            return "MEDIUM"  # Best balance
        elif laps_remaining > 15:
            return "SOFT"  # Quick stint to end
        else:
            return "HARD"  # Safe to finish
    
    # Strategy based on current compound and tire age
    if current == "SOFT":
        if tire_age > 15:  # SOFT worn after ~20 laps
            if laps_remaining > 25:
                return "MEDIUM"  # Need another stop
            else:
                return "HARD"  # Can finish on HARD
        elif tire_age > 10:
            return "MEDIUM"  # Prepare for next stop
        else:
            return "SOFT"  # Still good
    
    elif current == "MEDIUM":
        if tire_age > 35:  # MEDIUM can do 30-50 laps
            if laps_remaining > 15:
                return "MEDIUM"  # Fresh MEDIUM
            else:
                return "HARD"  # Safe to finish
        elif tire_age > 25:
            if laps_remaining < 20:
                return "HARD"  # One-stop to end
            else:
                return "MEDIUM"  # Fresh MEDIUM
        else:
            return "MEDIUM"  # Still optimal
    
    elif current == "HARD":
        # HARD can do full race at Monza 2024
        if tire_age > 40 and laps_remaining < 15:
            return "HARD"  # Push to finish
        elif tire_age > 35:
            if laps_remaining > 20:
                return "MEDIUM"  # Faster compound
            else:
                return "HARD"  # Finish on HARD
        else:
            return "HARD"  # Still good
    
    # Default: MEDIUM is safest choice for Monza 2024
    return "MEDIUM"

def generate_strategy_recommendation(
    scenario: str, 
    tire_condition: str, 
    confidence: float,
    laps_remaining: int,
    current_position: int,
    tire_age: int,
    compound: str
) -> Dict[str, Any]:
    """
    Generate comprehensive strategy recommendation for 2024 Monza.
    
    Context: Very low degradation, resurfaced track.
    Typical strategies: 1-stop (SOFT→HARD or MEDIUM→HARD) or 2-stop
    """
    
    recommendation = {
        "action": "",
        "priority": "",
        "reasoning": "",
        "laps_to_execute": None
    }
    
    # Critical scenarios
    if scenario == "PUNCTURE_OR_DAMAGE":
        recommendation["action"] = "PIT_IMMEDIATELY"
        recommendation["priority"] = "CRITICAL"
        recommendation["reasoning"] = "Suspected car damage or puncture detected"
        recommendation["laps_to_execute"] = 0
        return recommendation
    
    if scenario == "INLAP":
        recommendation["action"] = "PITTING_NOW"
        recommendation["priority"] = "HIGH"
        recommendation["reasoning"] = "Driver on inlap to pit lane"
        recommendation["laps_to_execute"] = 0
        return recommendation
    
    # Safety car opportunities
    if scenario == "SAFETY_CAR":
        if tire_condition in ["Worn", "Critical"] or tire_age > 20:
            recommendation["action"] = "PIT_UNDER_SAFETY_CAR"
            recommendation["priority"] = "HIGH"
            recommendation["reasoning"] = "Optimal time to pit - free stop under safety car"
            recommendation["laps_to_execute"] = 1
        else:
            recommendation["action"] = "HOLD_POSITION"
            recommendation["priority"] = "MEDIUM"
            recommendation["reasoning"] = "Tires still good, maintain track position"
        return recommendation
    
    # 2024 Monza specific: Compound-based strategy
    compound = compound.upper() if compound else "MEDIUM"
    
    if compound == "SOFT":
        if tire_age > 18:
            recommendation["action"] = "PIT_WITHIN_2_LAPS"
            recommendation["priority"] = "HIGH"
            recommendation["reasoning"] = "SOFT tires past optimal window (~20 laps at Monza)"
            recommendation["laps_to_execute"] = 2
        elif tire_age > 15:
            recommendation["action"] = "PLAN_PIT_STOP"
            recommendation["priority"] = "MEDIUM"
            recommendation["reasoning"] = "SOFT tires approaching wear limit"
            recommendation["laps_to_execute"] = 3
        else:
            recommendation["action"] = "PUSH_HARD"
            recommendation["priority"] = "LOW"
            recommendation["reasoning"] = "SOFT tires in optimal window, maximize pace"
    
    elif compound == "MEDIUM":
        if tire_age > 40:
            if laps_remaining < 15:
                recommendation["action"] = "MANAGE_TO_END"
                recommendation["priority"] = "MEDIUM"
                recommendation["reasoning"] = "MEDIUM can finish race (low degradation at Monza)"
            else:
                recommendation["action"] = "PIT_WITHIN_3_LAPS"
                recommendation["priority"] = "MEDIUM"
                recommendation["reasoning"] = "MEDIUM past 40 laps, consider fresher tires"
                recommendation["laps_to_execute"] = 3
        elif tire_age > 30:
            recommendation["action"] = "MONITOR_PACE"
            recommendation["priority"] = "LOW"
            recommendation["reasoning"] = "MEDIUM still strong (can do 30-50 laps)"
        else:
            recommendation["action"] = "MAINTAIN_PACE"
            recommendation["priority"] = "LOW"
            recommendation["reasoning"] = "MEDIUM in optimal window, continue current strategy"
    
    elif compound == "HARD":
        if tire_age > 45:
            if laps_remaining < 10:
                recommendation["action"] = "PUSH_TO_FINISH"
                recommendation["priority"] = "LOW"
                recommendation["reasoning"] = "HARD can complete full race distance at Monza"
            else:
                recommendation["action"] = "CONSIDER_UNDERCUT"
                recommendation["priority"] = "MEDIUM"
                recommendation["reasoning"] = "Long stint on HARD, consider fresher MEDIUM for pace"
                recommendation["laps_to_execute"] = 5
        else:
            recommendation["action"] = "MAINTAIN_PACE"
            recommendation["priority"] = "LOW"
            recommendation["reasoning"] = "HARD extremely durable at Monza 2024"
    
    # Tire condition override (if DL model detects issues)
    if tire_condition == "Critical" and confidence > 0.80:
        recommendation["action"] = "PIT_WITHIN_2_LAPS"
        recommendation["priority"] = "HIGH"
        recommendation["reasoning"] = "AI detected critical tire degradation"
        recommendation["laps_to_execute"] = 2
    
    return recommendation

# ----------------------------------------------------------------------
# 6. Main Analysis Pipeline
# ----------------------------------------------------------------------

class MonzaStrategyAnalyzer:
    """
    Complete analysis pipeline combining FastF1, ML, and DL.
    """
    
    def __init__(self, model_dir: str = "f1_monza_production_model", 
                 tire_model: str = "tire_degradation_model_final.h5"):
        self.pace_predictor = LapTimePredictor(model_dir)
        self.tire_classifier = TireDegradationClassifier(tire_model)
        self.data_fetcher = None
    
    def _build_ml_features(self, lap_features: Dict[str, Any], previous_laps: pd.DataFrame, driver: str) -> Dict[str, Any]:
        """Build complete feature set matching the training pipeline - using REAL data."""
        
        # Helper function to safely get numeric values
        def safe_get(key, default=0.0):
            val = lap_features.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)
        
        print(f"   → Building ML features using real telemetry data...")
        
        # Basic features
        features = {
            'LapNumber': int(safe_get('LapNumber', 1)),
            'Stint': int(safe_get('Stint', 1)),
            'TyreAge': int(safe_get('TyreLife', 0)),
            'Position': int(safe_get('Position', 10)),
        }
        
        # StintLap calculation
        features['StintLap'] = int(safe_get('TyreLife', 0))
        
        # REAL Speed features from telemetry
        features['MaxSpeed'] = safe_get('MaxSpeed', 340.0)
        features['SpeedStd'] = safe_get('SpeedStd', 15.0)
        
        # REAL Driver inputs from telemetry
        features['ThrottleMean'] = safe_get('ThrottleMean', 85.0)
        features['BrakeMean'] = safe_get('BrakeMean', 15.0)
        features['RpmMean'] = safe_get('RpmMean', 11500.0)
        
        # REAL Sector variance
        features['SectorVar'] = safe_get('SectorVar', 1.5)
        
        # Lag features
        features['PrevThrottle'] = features['ThrottleMean']  # Use same lap throttle
        features['LapTimeDelta'] = safe_get('LapTimeDelta', 0.0)
        features['SpeedDelta'] = safe_get('SpeedDelta', 0.0)
        
        # Rolling features from REAL data
        features['RollingAvg_3'] = safe_get('RollingAvg_3', safe_get('AvgPreviousLapTime', 85.0))
        features['RollingAvgSpeed_3'] = safe_get('RollingAvgSpeed_3', features['MaxSpeed'] * 0.85)
        features['RollingStd_3'] = safe_get('RollingStd_3', 0.5)
        
        # Engineered features using REAL data
        features['SpeedThrottleInteraction'] = features['MaxSpeed'] * features['ThrottleMean']
        
        # Normalized features
        features['NormLapNumber'] = min(features['LapNumber'] / 53.0, 1.0)
        features['NormTyreAge'] = min(features['TyreAge'] / 35.0, 1.0)
        
        # Position-based
        features['TireAgePosition'] = features['TyreAge'] * (1.0 / max(features['Position'], 1))
        features['StintEfficiency'] = features['StintLap'] / max(features['Stint'], 1)
        
        # Speed efficiency using REAL average speed
        avg_speed = safe_get('AvgSpeed', features['MaxSpeed'] * 0.85)
        est_lap_time = safe_get('AvgPreviousLapTime', 85.0)
        features['SpeedEfficiency'] = avg_speed / max(est_lap_time, 1.0)
        
        # Throttle/Brake ratio
        features['ThrottleBrakeRatio'] = features['ThrottleMean'] / max(features['BrakeMean'], 0.001)
        
        # Expected degradation based on 2024 Monza data
        compound_deg_rates = {
            'SOFT': 0.08,    # ~20 laps optimal (very low deg)
            'MEDIUM': 0.05,  # ~30-50 laps (extremely low deg)
            'HARD': 0.03,    # Can do full race (minimal deg)
            'INTERMEDIATE': 0.25,
            'WET': 0.30,
            'UNKNOWN': 0.06
        }
        compound = lap_features.get('Compound', 'HARD')
        if compound is None:
            compound = 'HARD'
        deg_rate = compound_deg_rates.get(str(compound), 0.12)
        features['ExpectedDegradation'] = features['TyreAge'] * deg_rate
        
        # Polynomial features
        features['TyreAge_sq'] = features['TyreAge'] ** 2
        features['LapNumber_sq'] = features['LapNumber'] ** 2
        features['Position_sq'] = features['Position'] ** 2
        
        # One-hot encode driver
        driver_num = lap_features.get('DriverNumber', '0')
        if driver_num:
            features[f'Driver_{driver_num}'] = 1.0
        
        # One-hot encode compound
        if compound:
            features[f'Compound_{compound}'] = 1.0
        
        # Ensure no NaN values
        for key, value in features.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                features[key] = 0.0
        
        print(f"   → ML features built: {len(features)} features ready for prediction")
        return features
    
    def analyze_driver_lap(self, year: int, driver: str, lap_number: int,
                          session_type: str = 'R') -> Dict[str, Any]:
        """
        Complete analysis for a specific driver lap.
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {driver} - Lap {lap_number} - Monza {year}")
        print(f"{'='*70}")
        
        # Load session data
        self.data_fetcher = MonzaDataFetcher(year, session_type)
        session = self.data_fetcher.load_session()
        
        if session is None:
            return {"error": "Failed to load session data"}
        
        # Get driver data
        all_driver_laps = self.data_fetcher.get_driver_lap_data(session, driver)
        if len(all_driver_laps) == 0:
            return {"error": f"No data found for driver {driver}"}
        
        # Get specific lap
        current_lap_df = all_driver_laps[all_driver_laps['LapNumber'] == lap_number]
        if len(current_lap_df) == 0:
            return {"error": f"Lap {lap_number} not found for {driver}"}
        
        current_lap = current_lap_df.iloc[0]
        previous_laps = all_driver_laps[all_driver_laps['LapNumber'] < lap_number]
        
        # Extract features
        lap_features = self.data_fetcher.extract_lap_features(current_lap, previous_laps)
        
        # Get lap time
        lap_time = lap_features['LapTime']
        if lap_time is None:
            return {"error": "No lap time recorded"}
        
        print(f"\n📊 Lap Time: {lap_time:.3f}s")
        print(f"🏎️  Compound: {lap_features['Compound']}")
        print(f"🔧 Tire Age: {lap_features['TyreLife']} laps")
        
        # ML Prediction
        print(f"\n🤖 STEP 1: Predicting optimal pace...")
        
        # Build comprehensive feature set matching training
        ml_features = self._build_ml_features(lap_features, previous_laps, driver)
        
        predicted_pace = self.pace_predictor.predict_pace(ml_features)
        print(f"   → Predicted optimal pace: {predicted_pace:.3f}s")
        print(f"   → Delta: {(lap_time - predicted_pace):.3f}s ({((lap_time - predicted_pace) / predicted_pace * 100):.1f}%)")
        
        # DL Tire Analysis
        print(f"\n🧠 STEP 2: Analyzing tire degradation...")
        
        # SMART OVERRIDE: Fresh tires should always be classified correctly
        tire_age = lap_features['TyreLife']
        compound = lap_features.get('Compound', 'UNKNOWN')
        
        # Override for brand new tires (first 5 laps on any compound)
        if tire_age <= 5:
            tire_analysis = {
                "condition": "Fresh",
                "confidence": 0.95,
                "probabilities": {
                    "Fresh": 0.95,
                    "Optimal": 0.04,
                    "Worn": 0.01,
                    "Critical": 0.0
                }
            }
            print(f"   → Tire Condition: Fresh (Override: {tire_age} laps old)")
            print(f"   → Confidence: 95.0%")
        # Use compound-specific thresholds for early classification
        elif compound == "SOFT" and tire_age <= 12:
            tire_analysis = {
                "condition": "Optimal",
                "confidence": 0.90,
                "probabilities": {
                    "Fresh": 0.05,
                    "Optimal": 0.90,
                    "Worn": 0.05,
                    "Critical": 0.0
                }
            }
            print(f"   → Tire Condition: Optimal (SOFT in prime window)")
            print(f"   → Confidence: 90.0%")
        elif compound == "MEDIUM" and tire_age <= 20:
            tire_analysis = {
                "condition": "Optimal",
                "confidence": 0.88,
                "probabilities": {
                    "Fresh": 0.08,
                    "Optimal": 0.88,
                    "Worn": 0.04,
                    "Critical": 0.0
                }
            }
            print(f"   → Tire Condition: Optimal (MEDIUM in prime window)")
            print(f"   → Confidence: 88.0%")
        elif compound == "HARD" and tire_age <= 25:
            tire_analysis = {
                "condition": "Optimal",
                "confidence": 0.85,
                "probabilities": {
                    "Fresh": 0.10,
                    "Optimal": 0.85,
                    "Worn": 0.05,
                    "Critical": 0.0
                }
            }
            print(f"   → Tire Condition: Optimal (HARD in prime window)")
            print(f"   → Confidence: 85.0%")
        # Only use DL model for tires with enough degradation history
        elif len(previous_laps) >= 10:
            recent_laps = previous_laps.tail(10)
            lap_times_for_dl = [t.total_seconds() for t in recent_laps['LapTime'] if pd.notna(t)]
            if len(lap_times_for_dl) >= 5:
                tire_analysis = self.tire_classifier.predict_state(lap_times_for_dl)
                print(f"   → Tire Condition: {tire_analysis['condition']} (DL Model)")
                print(f"   → Confidence: {tire_analysis['confidence']:.1%}")
            else:
                tire_analysis = {"condition": "Optimal", "confidence": 0.80, 
                               "probabilities": {"Fresh": 0.1, "Optimal": 0.8, "Worn": 0.1, "Critical": 0.0}}
                print(f"   → Tire Condition: Optimal (Insufficient data)")
                print(f"   → Confidence: 80.0%")
        else:
            tire_analysis = {"condition": "Optimal", "confidence": 0.85, 
                           "probabilities": {"Fresh": 0.1, "Optimal": 0.85, "Worn": 0.05, "Critical": 0.0}}
            print(f"   → Tire Condition: Optimal (Early stint)")
            print(f"   → Confidence: 85.0%")
        
        # Scenario Detection
        print(f"\n🔍 STEP 3: Detecting lap scenario...")
        scenario, analysis_details = detect_lap_scenario(lap_features, predicted_pace, previous_laps)
        print(f"   → Scenario: {scenario}")
        
        # Strategy Recommendation
        print(f"\n⚡ STEP 4: Generating strategy recommendation...")
        total_laps = session.laps['LapNumber'].max()
        laps_remaining = total_laps - lap_number
        strategy = generate_strategy_recommendation(
            scenario, 
            tire_analysis['condition'],
            tire_analysis['confidence'],
            laps_remaining,
            current_position=1,
            tire_age=lap_features['TyreLife'],
            compound=lap_features['Compound']
        )
        print(f"   → Action: {strategy['action']}")
        print(f"   → Priority: {strategy['priority']}")
        print(f"   → Reasoning: {strategy['reasoning']}")
        
        # Compound Recommendation
        print(f"\n🔧 STEP 5: Tire compound recommendation...")
        next_compound = recommend_compound(
            lap_features['Compound'], 
            scenario,
            tire_analysis['condition'],
            lap_features['TyreLife'],
            laps_remaining
        )
        print(f"   → Suggested next compound: {next_compound}")
        
        # Compile results for frontend
        result = {
            "metadata": {
                "driver": driver,
                "lap_number": lap_number,
                "year": year,
                "circuit": "Monza",
                "session_type": session_type,
                "timestamp": datetime.now().isoformat()
            },
            "lap_data": {
                "actual_lap_time": round(lap_time, 3),
                "predicted_pace": round(predicted_pace, 3),
                "delta_seconds": round(lap_time - predicted_pace, 3),
                "delta_percent": round((lap_time - predicted_pace) / predicted_pace * 100, 2),
                "compound": lap_features['Compound'],
                "tire_age": lap_features['TyreLife'],
                "stint": lap_features['Stint']
            },
            "tire_analysis": {
                "condition": tire_analysis['condition'],
                "confidence": round(tire_analysis['confidence'], 3),
                "probabilities": {k: round(v, 3) for k, v in tire_analysis['probabilities'].items()}
            },
            "scenario": {
                "classification": scenario,
                "details": analysis_details
            },
            "strategy": strategy,
            "recommendation": {
                "next_compound": next_compound
            },
            "race_context": {
                "laps_remaining": laps_remaining,
                "total_laps": int(total_laps)
            }
        }
        
        print(f"\n{'='*70}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*70}\n")
        
        return result

# ----------------------------------------------------------------------
# 7. Export for Frontend
# ----------------------------------------------------------------------

def save_analysis_for_frontend(result: Dict[str, Any], output_file: str = "monza_analysis.json"):
    """Save analysis results as JSON for frontend consumption."""
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"💾 Analysis saved to {output_file}")

# ----------------------------------------------------------------------
# 8. Main Execution (Only runs if script is executed directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # This only runs when you execute: python monza_strategy_backend.py
    # It will NOT run when imported by api_server.py
    
    print("\n" + "="*70)
    print("STANDALONE MODE - Running Example Analysis")
    print("="*70)
    
    # Initialize analyzer
    analyzer = MonzaStrategyAnalyzer()
    
    # Example: Analyze Verstappen's lap 25 from 2024 Monza race
    result = analyzer.analyze_driver_lap(
        year=2024,
        driver='VER',
        lap_number=25,
        session_type='R'
    )
    
    # Save for frontend
    if "error" not in result:
        save_analysis_for_frontend(result)
        
        # Print summary
        print("\n📱 FRONTEND DISPLAY SUMMARY:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Error: {result['error']}")