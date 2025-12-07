
# ============================================================================
# TIRE DEGRADATION PREDICTION MODEL - DEPLOYMENT PACKAGE
# ============================================================================
# Version: 1.0
# Accuracy: 85.95%
# Trained on: 2023-2024 F1 Season Data
# Model: CNN-LSTM Hybrid
# ============================================================================

import numpy as np
import tensorflow as tf

class TireDegradationPredictor:
    """Production-ready tire degradation predictor"""

    def __init__(self, model_path='best_tire_model.h5'):
        """Initialize predictor with trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.labels = ['Fresh', 'Optimal', 'Worn', 'Critical']

    def predict(self, lap_times):
        """
        Predict tire condition from lap times

        Args:
            lap_times: List of 10 consecutive lap times in seconds

        Returns:
            dict: Prediction results with confidence and recommendations
        """
        # Validate input
        if len(lap_times) != 10:
            raise ValueError(f"Expected 10 lap times, got {len(lap_times)}")

        # Prepare input
        input_seq = np.array(lap_times, dtype=np.float32).reshape(1, 10, 1)

        # Predict
        predictions = self.model.predict(input_seq, verbose=0)

        # Parse results
        condition_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return {
            'condition': self.labels[condition_idx],
            'confidence': confidence,
            'condition_idx': int(condition_idx),
            'probabilities': {
                self.labels[i]: float(pred) 
                for i, pred in enumerate(predictions[0].flatten())
            },
            'recommendation': self._get_recommendation(condition_idx, confidence)
        }

    def _get_recommendation(self, condition_idx, confidence):
        """Get strategy recommendation based on condition and confidence"""
        recommendations = {
            0: "Push hard - tires in optimal condition",
            1: "Maintain pace - monitor for pit window",
            2: "Consider pit stop within 3-5 laps",
            3: "BOX BOX BOX - immediate pit required"
        }

        if confidence < 0.6:
            return f"Low confidence: {recommendations[condition_idx]}"
        elif confidence < 0.8:
            return f"Medium confidence: {recommendations[condition_idx]}"
        else:
            return f"High confidence: {recommendations[condition_idx]}"

    def batch_predict(self, sequences):
        """
        Predict multiple sequences at once

        Args:
            sequences: List of sequences, each with 10 lap times

        Returns:
            list: Prediction results for each sequence
        """
        return [self.predict(seq) for seq in sequences]

# ============================================================================
# USAGE EXAMPLE:
# ============================================================================
if __name__ == "__main__":
    # Initialize predictor
    predictor = TireDegradationPredictor()

    # Example lap times (10 consecutive laps)
    lap_times = [85.0, 85.2, 85.5, 85.9, 86.4, 87.0, 87.7, 88.5, 89.4, 90.4]

    # Make prediction
    result = predictor.predict(lap_times)

    print(f"Condition: {result['condition']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
