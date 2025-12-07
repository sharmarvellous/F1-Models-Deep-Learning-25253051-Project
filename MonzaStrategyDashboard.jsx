import React, { useState } from 'react';
import { AlertCircle, TrendingUp, TrendingDown, Clock, Gauge, Settings, Flag, AlertTriangle } from 'lucide-react';

const MonzaStrategyDashboard = () => {
  // Example data - replace with actual API call to your Python backend
  const [analysisData] = useState({
    "metadata": {
      "driver": "VER",
      "lap_number": 25,
      "year": 2024,
      "circuit": "Monza",
      "session_type": "R"
    },
    "lap_data": {
      "actual_lap_time": 85.342,
      "predicted_pace": 83.120,
      "delta_seconds": 2.222,
      "delta_percent": 2.67,
      "compound": "MEDIUM",
      "tire_age": 12,
      "stint": 2
    },
    "tire_analysis": {
      "condition": "Optimal",
      "confidence": 0.892,
      "probabilities": {
        "Fresh": 0.05,
        "Optimal": 0.89,
        "Worn": 0.05,
        "Critical": 0.01
      }
    },
    "scenario": {
      "classification": "TRAFFIC",
      "details": {
        "pace_delta_pct": 2.67,
        "speed_delta": -3.2
      }
    },
    "strategy": {
      "action": "MAINTAIN_PACE",
      "priority": "LOW",
      "reasoning": "Continue current strategy",
      "laps_to_execute": null
    },
    "recommendation": {
      "next_compound": "MEDIUM"
    },
    "race_context": {
      "laps_remaining": 28,
      "total_laps": 53
    }
  });

  // Color mappings
  const priorityColors = {
    CRITICAL: 'bg-red-600',
    HIGH: 'bg-orange-500',
    MEDIUM: 'bg-yellow-500',
    LOW: 'bg-green-500'
  };

  const tireConditionColors = {
    Fresh: 'bg-green-500',
    Optimal: 'bg-blue-500',
    Worn: 'bg-orange-500',
    Critical: 'bg-red-500'
  };

  const compoundColors = {
    SOFT: 'bg-red-500',
    MEDIUM: 'bg-yellow-400',
    HARD: 'bg-gray-300 text-gray-900'
  };

  const scenarioIcons = {
    TRAFFIC: AlertCircle,
    NORMAL_PACE: TrendingUp,
    STRONG_PACE: TrendingUp,
    INLAP: Flag,
    PIT_OUTLAP_OR_INLAP: Settings,
    PUNCTURE_OR_DAMAGE: AlertTriangle,
    SAFETY_CAR: AlertCircle
  };

  const ScenarioIcon = scenarioIcons[analysisData.scenario.classification] || AlertCircle;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-red-500 to-red-700 bg-clip-text text-transparent">
              🏎️ Monza Strategy Analysis
            </h1>
            <p className="text-gray-400">
              {analysisData.metadata.driver} • Lap {analysisData.metadata.lap_number} of {analysisData.race_context.total_laps} • {analysisData.metadata.year}
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{analysisData.lap_data.actual_lap_time.toFixed(3)}s</div>
            <div className={`text-sm ${analysisData.lap_data.delta_percent > 0 ? 'text-red-400' : 'text-green-400'}`}>
              {analysisData.lap_data.delta_percent > 0 ? '+' : ''}{analysisData.lap_data.delta_seconds.toFixed(3)}s
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Lap Performance Card */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
          <div className="flex items-center mb-4">
            <Clock className="mr-2 text-blue-400" size={24} />
            <h2 className="text-xl font-bold">Lap Performance</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Actual Lap Time</span>
                <span className="font-mono">{analysisData.lap_data.actual_lap_time.toFixed(3)}s</span>
              </div>
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Predicted Pace</span>
                <span className="font-mono">{analysisData.lap_data.predicted_pace.toFixed(3)}s</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                <div 
                  className={`h-2 rounded-full ${analysisData.lap_data.delta_percent > 3 ? 'bg-red-500' : analysisData.lap_data.delta_percent > 1 ? 'bg-yellow-500' : 'bg-green-500'}`}
                  style={{ width: `${Math.min(100, Math.abs(analysisData.lap_data.delta_percent) * 10)}%` }}
                />
              </div>
              <div className="text-center text-xs text-gray-500 mt-1">
                Delta: {analysisData.lap_data.delta_percent.toFixed(2)}%
              </div>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-900 rounded p-3">
                  <div className="text-xs text-gray-400 mb-1">Compound</div>
                  <div className={`inline-block px-3 py-1 rounded font-bold text-sm ${compoundColors[analysisData.lap_data.compound]}`}>
                    {analysisData.lap_data.compound}
                  </div>
                </div>
                <div className="bg-gray-900 rounded p-3">
                  <div className="text-xs text-gray-400 mb-1">Tire Age</div>
                  <div className="text-2xl font-bold">{analysisData.lap_data.tire_age}</div>
                  <div className="text-xs text-gray-500">laps</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tire Analysis Card */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
          <div className="flex items-center mb-4">
            <Gauge className="mr-2 text-purple-400" size={24} />
            <h2 className="text-xl font-bold">Tire Condition</h2>
          </div>
          
          <div className="text-center mb-6">
            <div className={`inline-block px-6 py-3 rounded-lg font-bold text-2xl ${tireConditionColors[analysisData.tire_analysis.condition]}`}>
              {analysisData.tire_analysis.condition}
            </div>
            <div className="text-sm text-gray-400 mt-2">
              Confidence: {(analysisData.tire_analysis.confidence * 100).toFixed(1)}%
            </div>
          </div>

          <div className="space-y-3">
            {Object.entries(analysisData.tire_analysis.probabilities).map(([condition, prob]) => (
              <div key={condition}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">{condition}</span>
                  <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${tireConditionColors[condition]}`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Scenario & Strategy Card */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
          <div className="flex items-center mb-4">
            <ScenarioIcon className="mr-2 text-yellow-400" size={24} />
            <h2 className="text-xl font-bold">Scenario Analysis</h2>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <div className="text-sm text-gray-400 mb-1">Detected Scenario</div>
            <div className="text-xl font-bold text-yellow-400 mb-2">
              {analysisData.scenario.classification.replace(/_/g, ' ')}
            </div>
            {analysisData.scenario.details.pace_delta_pct && (
              <div className="text-xs text-gray-500">
                Pace Delta: {analysisData.scenario.details.pace_delta_pct.toFixed(2)}%
              </div>
            )}
          </div>

          <div className={`rounded-lg p-4 mb-4 ${priorityColors[analysisData.strategy.priority]} bg-opacity-20 border-2`}
               style={{ borderColor: `var(--${analysisData.strategy.priority.toLowerCase()})` }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold">Priority</span>
              <span className={`px-2 py-1 rounded text-xs font-bold ${priorityColors[analysisData.strategy.priority]}`}>
                {analysisData.strategy.priority}
              </span>
            </div>
            <div className="text-lg font-bold mb-2">{analysisData.strategy.action.replace(/_/g, ' ')}</div>
            <div className="text-sm text-gray-300">{analysisData.strategy.reasoning}</div>
            {analysisData.strategy.laps_to_execute && (
              <div className="text-xs text-gray-400 mt-2">
                Execute within {analysisData.strategy.laps_to_execute} laps
              </div>
            )}
          </div>

          <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Next Compound Recommendation</div>
            <div className={`inline-block px-4 py-2 rounded-lg font-bold ${compoundColors[analysisData.recommendation.next_compound]}`}>
              {analysisData.recommendation.next_compound}
            </div>
          </div>
        </div>
      </div>

      {/* Race Context Footer */}
      <div className="max-w-7xl mx-auto mt-6">
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div>
              <div className="text-xs text-gray-400">Laps Remaining</div>
              <div className="text-2xl font-bold">{analysisData.race_context.laps_remaining}</div>
            </div>
            <div className="h-10 w-px bg-gray-700" />
            <div>
              <div className="text-xs text-gray-400">Current Stint</div>
              <div className="text-2xl font-bold">{analysisData.lap_data.stint}</div>
            </div>
            <div className="h-10 w-px bg-gray-700" />
            <div>
              <div className="text-xs text-gray-400">Progress</div>
              <div className="text-lg font-bold">
                {((analysisData.metadata.lap_number / analysisData.race_context.total_laps) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          
          <div className="w-1/3">
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-blue-500 to-red-500 h-3 rounded-full transition-all"
                style={{ width: `${(analysisData.metadata.lap_number / analysisData.race_context.total_laps) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* API Integration Note */}
      <div className="max-w-7xl mx-auto mt-6 text-center text-gray-500 text-sm">
        <p>💡 Connect this frontend to your Python backend API to load real-time FastF1 data</p>
      </div>
    </div>
  );
};

export default MonzaStrategyDashboard;