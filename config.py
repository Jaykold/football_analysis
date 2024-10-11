import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')