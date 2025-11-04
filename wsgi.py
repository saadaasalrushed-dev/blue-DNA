"""
Blue DNA - AI Beach Guardian
WSGI Configuration for PythonAnywhere
"""

import sys
import os

# Add the project directory to Python path
# UPDATE THIS PATH: Your files will be in /home/AlRushedAPSch/mysite/
project_home = '/home/AlRushedAPSch/mysite'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change to project directory
os.chdir(project_home)

# Import Flask app
from app import app as application

# For PythonAnywhere, the WSGI file must be named 'application'
# Do not change this variable name!

