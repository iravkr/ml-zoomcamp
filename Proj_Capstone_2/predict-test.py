import requests

# Define the URL of the prediction endpoint
url = 'http://localhost:9696/predict'

# Define customer data with values (assuming data has been encoded and scaled)
# Define customer data with values (assuming data has been encoded and scaled)
customer = {
    "Temperature": 25.5,
    "Humidity": 45.2,
    "SquareFootage": 1500,
    "Occupancy": 4,
    "HVACUsage": "On",
    "LightingUsage": "Off",
    "RenewableEnergy": 3.5,
    "DayOfWeek": "Monday",
    "Holiday": "No"
}


# Send a POST request to the prediction endpoint and get the response
response = requests.post(url, json=customer).json()

# Print the response
print(response)
