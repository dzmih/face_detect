# useing python 3.13 cuz its what we have
from python:3.13-slim

# install sys dependencies for opencv and gui
# hope i didnt miss any libs lol
run apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# set work dir
workdir /app

# copy requirements first to cache them
copy requirements.txt .
run pip install --no-cache-dir -r requirements.txt

# copy the rest of the code
copy . .

# command to run the app
cmd ["python", "main.py"]