"""
Script to install dependencies for the churn prediction system.
"""

import os
import sys
import logging
import subprocess
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_virtual_environment():
    """Create a virtual environment."""
    try:
        logger.info("Creating virtual environment...")
        subprocess.run(
            "python -m venv .venv",
            shell=True,
            check=True
        )
        logger.info("Virtual environment created successfully")
    except Exception as e:
        logger.error(f"Error creating virtual environment: {str(e)}")
        raise

def install_python_packages():
    """Install Python packages from requirements.txt."""
    try:
        logger.info("Installing Python packages...")
        
        # Determine the pip command based on the platform
        if platform.system() == "Windows":
            pip_cmd = ".venv\\Scripts\\pip"
        else:
            pip_cmd = ".venv/bin/pip"
        
        # Upgrade pip
        subprocess.run(
            f"{pip_cmd} install --upgrade pip",
            shell=True,
            check=True
        )
        
        # Install packages
        subprocess.run(
            f"{pip_cmd} install -r requirements.txt",
            shell=True,
            check=True
        )
        
        logger.info("Python packages installed successfully")
    except Exception as e:
        logger.error(f"Error installing Python packages: {str(e)}")
        raise

def install_system_dependencies():
    """Install system dependencies."""
    try:
        system = platform.system()
        
        if system == "Linux":
            # Install system packages on Linux
            logger.info("Installing system dependencies on Linux...")
            subprocess.run(
                "sudo apt-get update && sudo apt-get install -y python3-dev build-essential",
                shell=True,
                check=True
            )
            
        elif system == "Darwin":  # macOS
            # Install system packages on macOS
            logger.info("Installing system dependencies on macOS...")
            subprocess.run(
                "brew install python3",
                shell=True,
                check=True
            )
            
        elif system == "Windows":
            # Windows dependencies are handled by pip
            logger.info("No additional system dependencies required for Windows")
            
        else:
            logger.warning(f"Unsupported operating system: {system}")
            
    except Exception as e:
        logger.error(f"Error installing system dependencies: {str(e)}")
        raise

def install_mlflow():
    """Install MLflow."""
    try:
        logger.info("Installing MLflow...")
        subprocess.run(
            "pip install mlflow",
            shell=True,
            check=True
        )
        logger.info("MLflow installed successfully")
    except Exception as e:
        logger.error(f"Error installing MLflow: {str(e)}")
        raise

def install_monitoring_tools():
    """Install monitoring tools (Prometheus and Grafana)."""
    try:
        system = platform.system()
        
        if system == "Linux":
            # Install Prometheus
            logger.info("Installing Prometheus...")
            subprocess.run(
                "sudo apt-get install -y prometheus",
                shell=True,
                check=True
            )
            
            # Install Grafana
            logger.info("Installing Grafana...")
            subprocess.run(
                "sudo apt-get install -y grafana",
                shell=True,
                check=True
            )
            
        elif system == "Darwin":  # macOS
            # Install Prometheus
            logger.info("Installing Prometheus...")
            subprocess.run(
                "brew install prometheus",
                shell=True,
                check=True
            )
            
            # Install Grafana
            logger.info("Installing Grafana...")
            subprocess.run(
                "brew install grafana",
                shell=True,
                check=True
            )
            
        elif system == "Windows":
            logger.warning("Please install Prometheus and Grafana manually on Windows")
            
        else:
            logger.warning(f"Unsupported operating system: {system}")
            
    except Exception as e:
        logger.error(f"Error installing monitoring tools: {str(e)}")
        raise

def install_dependencies():
    """Install all dependencies."""
    try:
        # Create virtual environment
        create_virtual_environment()
        
        # Install system dependencies
        install_system_dependencies()
        
        # Install Python packages
        install_python_packages()
        
        # Install MLflow
        install_mlflow()
        
        # Install monitoring tools
        install_monitoring_tools()
        
        logger.info("All dependencies installed successfully")
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        raise

if __name__ == "__main__":
    install_dependencies() 