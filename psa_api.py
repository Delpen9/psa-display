import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variables
token = os.getenv("PSA_API_TOKEN")

# Prepare the API request
url = "https://api.psacard.com/publicapi/cert/GetByCertNumber/00000000"
headers = {
    "authorization": f"bearer {token}"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Print the JSON response
print(response.json())
