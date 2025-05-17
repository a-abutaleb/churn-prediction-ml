"""
Script to run the monitoring system for the churn prediction model.
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_prometheus():
    """Start Prometheus server."""
    try:
        # Create Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'churn_prediction'
    static_configs:
      - targets: ['localhost:8000']
        """
        
        # Write configuration to file
        os.makedirs("monitoring/prometheus", exist_ok=True)
        with open("monitoring/prometheus/prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        # Start Prometheus
        logger.info("Starting Prometheus server...")
        prometheus = subprocess.Popen(
            "prometheus --config.file=monitoring/prometheus/prometheus.yml",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return prometheus
        
    except Exception as e:
        logger.error(f"Error starting Prometheus: {str(e)}")
        raise

def start_grafana():
    """Start Grafana server."""
    try:
        # Create Grafana configuration
        grafana_config = """
[server]
http_port = 3000
root_url = http://localhost:3000/

[security]
admin_user = admin
admin_password = admin

[auth.anonymous]
enabled = true
        """
        
        # Write configuration to file
        os.makedirs("monitoring/grafana", exist_ok=True)
        with open("monitoring/grafana/grafana.ini", "w") as f:
            f.write(grafana_config)
        
        # Start Grafana
        logger.info("Starting Grafana server...")
        grafana = subprocess.Popen(
            "grafana-server --config monitoring/grafana/grafana.ini",
            shell=True
        )
        time.sleep(5)  # Wait for server to start
        return grafana
        
    except Exception as e:
        logger.error(f"Error starting Grafana: {str(e)}")
        raise

def start_monitoring():
    """Start the monitoring system."""
    try:
        # Start Prometheus
        prometheus = start_prometheus()
        
        # Start Grafana
        grafana = start_grafana()
        
        # Start drift monitoring
        logger.info("Starting drift monitoring...")
        monitor = subprocess.Popen(
            "python src/monitoring/drift/monitor.py",
            shell=True
        )
        
        # Keep the script running
        try:
            while True:
                time.sleep(60)
                logger.info("Monitoring system is running...")
        except KeyboardInterrupt:
            logger.info("Stopping monitoring system...")
            prometheus.terminate()
            grafana.terminate()
            monitor.terminate()
            logger.info("Monitoring system stopped")
            
    except Exception as e:
        logger.error(f"Error in monitoring system: {str(e)}")
        raise

if __name__ == "__main__":
    start_monitoring() 