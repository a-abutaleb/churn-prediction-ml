import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from flask import Flask, render_template
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_monitoring_data(model_name: str) -> pd.DataFrame:
    """Load monitoring data from JSON file."""
    try:
        with open(f'monitoring/{model_name}_monitoring.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading monitoring data: {str(e)}")
        return pd.DataFrame()

def create_performance_plot(df: pd.DataFrame) -> go.Figure:
    """Create performance metrics plot."""
    if df.empty:
        return go.Figure()
    
    # Extract performance metrics
    metrics_df = pd.DataFrame([
        {**row['performance_metrics'], 'timestamp': row['timestamp']}
        for row in df.to_dict('records')
        if 'performance_metrics' in row
    ])
    
    if metrics_df.empty:
        return go.Figure()
    
    # Create line plot
    fig = go.Figure()
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df[metric],
                name=metric.capitalize(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Model Performance Metrics Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Score',
        yaxis_range=[0, 1]
    )
    
    return fig

def create_drift_plot(df: pd.DataFrame) -> go.Figure:
    """Create drift detection plot."""
    if df.empty:
        return go.Figure()
    
    # Count drifted features over time
    drift_counts = []
    for _, row in df.iterrows():
        if 'drift_results' in row:
            drifted_features = [
                feature for feature, results in row['drift_results'].items()
                if results.get('is_drift', False)
            ]
            drift_counts.append({
                'timestamp': row['timestamp'],
                'drifted_features_count': len(drifted_features)
            })
    
    drift_df = pd.DataFrame(drift_counts)
    
    if drift_df.empty:
        return go.Figure()
    
    # Create bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drift_df['timestamp'],
        y=drift_df['drifted_features_count'],
        name='Drifted Features'
    ))
    
    fig.update_layout(
        title='Number of Features with Drift Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Number of Drifted Features'
    )
    
    return fig

@app.route('/')
def dashboard():
    """Render monitoring dashboard."""
    try:
        # Load monitoring data
        df = load_monitoring_data('churn_prediction_xgboost')
        
        if df.empty:
            return render_template('dashboard.html', 
                                performance_plot=None,
                                drift_plot=None,
                                error="No monitoring data available")
        
        # Create plots
        performance_plot = create_performance_plot(df)
        drift_plot = create_drift_plot(df)
        
        # Get latest metrics
        latest_metrics = None
        if 'performance_metrics' in df.iloc[-1]:
            latest_metrics = df.iloc[-1]['performance_metrics']
        
        # Get drift summary
        drift_summary = None
        if 'drift_results' in df.iloc[-1]:
            drifted_features = [
                feature for feature, results in df.iloc[-1]['drift_results'].items()
                if results.get('is_drift', False)
            ]
            drift_summary = {
                'drifted_features': drifted_features,
                'total_features': len(df.iloc[-1]['drift_results'])
            }
        
        return render_template('dashboard.html',
                             performance_plot=performance_plot.to_html(full_html=False),
                             drift_plot=drift_plot.to_html(full_html=False),
                             latest_metrics=latest_metrics,
                             drift_summary=drift_summary)
    
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return render_template('dashboard.html',
                             error=str(e))

def main():
    """Run the monitoring dashboard."""
    try:
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # Create dashboard template
        with open('templates/dashboard.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Model Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .plot { margin-bottom: 30px; }
        .metrics { margin-bottom: 20px; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Monitoring Dashboard</h1>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% else %}
        
        <div class="metrics">
            <h2>Latest Performance Metrics</h2>
            {% if latest_metrics %}
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for metric, value in latest_metrics.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td>{{ "%.3f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No performance metrics available</p>
            {% endif %}
        </div>
        
        <div class="metrics">
            <h2>Drift Summary</h2>
            {% if drift_summary %}
            <p>Total Features: {{ drift_summary.total_features }}</p>
            <p>Drifted Features: {{ drift_summary.drifted_features|length }}</p>
            {% if drift_summary.drifted_features %}
            <h3>Drifted Features:</h3>
            <ul>
                {% for feature in drift_summary.drifted_features %}
                <li>{{ feature }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            {% else %}
            <p>No drift information available</p>
            {% endif %}
        </div>
        
        <div class="plot">
            <h2>Performance Metrics Over Time</h2>
            {{ performance_plot|safe }}
        </div>
        
        <div class="plot">
            <h2>Data Drift Over Time</h2>
            {{ drift_plot|safe }}
        </div>
        
        {% endif %}
    </div>
</body>
</html>
            ''')
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5002)
    
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

if __name__ == '__main__':
    main() 