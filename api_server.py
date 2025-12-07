from flask import Flask, jsonify, request
from flask_cors import CORS
from monza_strategy_backend import MonzaStrategyAnalyzer
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global analyzer instance (lazy initialization)
analyzer = None

def get_analyzer():
    """Lazy initialization of analyzer to handle startup errors gracefully."""
    global analyzer
    if analyzer is None:
        try:
            analyzer = MonzaStrategyAnalyzer(
                model_dir="f1_monza_production_model",
                tire_model="tire_degradation_model_final.h5"
            )
            print("✅ Analyzer initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing analyzer: {e}")
            raise
    return analyzer

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    try:
        analyzer_status = "ready" if analyzer is not None else "not_initialized"
        return jsonify({
            "status": "healthy",
            "analyzer": analyzer_status
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_lap():
    """
    Analyze a specific lap for a driver.
    
    Expected JSON body:
    {
        "driver": "VER",
        "lap_number": 25,
        "year": 2024,
        "session_type": "R"
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        data = request.json
        
        # Validate required fields
        required_fields = ['driver', 'lap_number']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Get analyzer instance
        try:
            current_analyzer = get_analyzer()
        except Exception as e:
            return jsonify({
                "error": "Failed to initialize analyzer",
                "details": str(e)
            }), 500
        
        # Extract parameters
        driver = data['driver'].upper()
        lap_number = int(data['lap_number'])
        year = data.get('year', 2024)
        session_type = data.get('session_type', 'R')
        
        # Validate parameters
        if lap_number < 1:
            return jsonify({
                "error": "lap_number must be positive"
            }), 400
        
        if len(driver) < 3:
            return jsonify({
                "error": "driver code must be at least 3 characters (e.g., 'VER', 'HAM')"
            }), 400
        
        # Run analysis
        print(f"📊 Analyzing {driver} lap {lap_number} from {year} Monza...")
        try:
            result = current_analyzer.analyze_driver_lap(
                year=year,
                driver=driver,
                lap_number=lap_number,
                session_type=session_type
            )
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Analysis error: {error_msg}")
            traceback.print_exc()
            return jsonify({
                "error": "Analysis failed",
                "details": error_msg,
                "driver": driver,
                "lap": lap_number
            }), 500
        
        # Check if analysis returned an error
        if "error" in result:
            return jsonify(result), 404
        
        print(f"✅ Analysis complete for {driver}")
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid input format",
            "details": str(e)
        }), 400
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error during analysis",
            "details": str(e)
        }), 500

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    """
    Get list of available drivers for a specific year and session.
    
    Query parameters:
    - year: Race year (default: 2024)
    - session_type: Session type (default: 'R')
    """
    try:
        year = int(request.args.get('year', 2024))
        session_type = request.args.get('session_type', 'R')
        
        from monza_strategy_backend import MonzaDataFetcher
        fetcher = MonzaDataFetcher(year, session_type)
        session = fetcher.load_session()
        
        if session is None:
            return jsonify({
                "error": "Failed to load session data"
            }), 500
        
        # Get unique drivers
        drivers = session.laps['Driver'].unique().tolist()
        
        return jsonify({
            "year": year,
            "session_type": session_type,
            "drivers": drivers,
            "total_laps": int(session.laps['LapNumber'].max())
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Failed to fetch drivers",
            "details": str(e)
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple laps at once.
    
    Expected JSON body:
    {
        "analyses": [
            {"driver": "VER", "lap_number": 25, "year": 2024},
            {"driver": "HAM", "lap_number": 25, "year": 2024}
        ]
    }
    """
    try:
        if not request.json or 'analyses' not in request.json:
            return jsonify({
                "error": "Missing 'analyses' array in request"
            }), 400
        
        analyses = request.json['analyses']
        
        if not isinstance(analyses, list) or len(analyses) == 0:
            return jsonify({
                "error": "'analyses' must be a non-empty array"
            }), 400
        
        # Limit batch size to prevent overload
        if len(analyses) > 10:
            return jsonify({
                "error": "Maximum 10 analyses per batch request"
            }), 400
        
        current_analyzer = get_analyzer()
        results = []
        
        for idx, analysis_req in enumerate(analyses):
            try:
                result = current_analyzer.analyze_driver_lap(
                    year=analysis_req.get('year', 2024),
                    driver=analysis_req['driver'].upper(),
                    lap_number=int(analysis_req['lap_number']),
                    session_type=analysis_req.get('session_type', 'R')
                )
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        return jsonify({
            "total": len(analyses),
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Batch analysis failed",
            "details": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    print("="*70)
    print("🏎️  MONZA STRATEGY API SERVER")
    print("="*70)
    print("\nAvailable endpoints:")
    print("  GET  /api/health          - Health check")
    print("  POST /api/analyze         - Analyze single lap")
    print("  GET  /api/drivers         - Get available drivers")
    print("  POST /api/batch-analyze   - Analyze multiple laps")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True,
        use_reloader=False  # Prevent double loading
    )