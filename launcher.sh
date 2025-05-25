#!/bin/bash

# Step 1: Go to the script's directory (project root)
cd "$(dirname "$0")"

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Step 3: Activate virtual environment
source venv/bin/activate

# Step 4: Install requirements
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "No requirements.txt found, skipping pip install."
fi

# Step 5: Run Streamlit
streamlit run display.py
