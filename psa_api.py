import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get the token from environment variables
    token = os.getenv("PSA_API_TOKEN")

    specID = "2853456"
    certNum = "55613523"

    # # Prepare the API request
    # url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{certNum}"
    # headers = {
    #     "authorization": f"bearer {token}"
    # }

    # # Make the GET request
    # response = requests.get(url, headers=headers)

    # # Print the JSON response
    # print(response.json())

    # Prepare the API request
    url = f"https://api.psacard.com/publicapi/pop/GetPSASpecPopulation/{specID}"
    headers = {
        "authorization": f"bearer {token}"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Print the JSON response
    print(response.json())
