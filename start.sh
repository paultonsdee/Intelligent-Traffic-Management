#!/bin/bash

# Check if Python 3.10 is installed
if ! python3.10 --version &> /dev/null; then
  echo "Python 3.10 is not installed. Installing Python 3.10..."
  if [ "$(uname)" == "Darwin" ]; then
    # macOS installation
    brew install python@3.10
  elif [ -f /etc/debian_version ]; then
    # Debian/Ubuntu installation
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv
  elif [ -f /etc/redhat-release ]; then
    # RHEL/CentOS/Fedora installation
    sudo yum install -y python3.10
  else
    echo "Unsupported OS. Please install Python 3.10 manually."
    exit 1
  fi
fi

# Check if venv directory exists, create if not
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3.10 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install required libraries
echo "Installing required libraries..."
pip install --upgrade pip
pip install -r requirements.txt

# Run configuration setup
echo "Setting up the configuration..."
python config/config_private.py

# Run the main application
echo "Running the project..."
python main.py
