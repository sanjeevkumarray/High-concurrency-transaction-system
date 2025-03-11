import os
import joblib
import pandas as pd
from flask import flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
