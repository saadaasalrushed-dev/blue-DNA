"""
Blue DNA - AI Beach Guardian
WSGI Configuration for PythonAnywhere
SPECIFIC FOR USER: AlRushedAPSch
"""

import sys

# Add the project directory to Python path
path = '/home/AlRushedAPSch/mysite'
if path not in sys.path:
    sys.path.insert(0, path)

# Change to project directory
import os
os.chdir(path)

# Import Flask app
from app import app as application

# For PythonAnywhere, the WSGI file must be named 'application'
# Do not change this variable name!

