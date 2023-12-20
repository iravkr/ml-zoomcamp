import requests

# Define the URL of the prediction endpoint
url = 'http://localhost:9696/predict'

# Define customer data with values (assuming data has been encoded and scaled)
# Define customer data with values (assuming data has been encoded and scaled)
customer = {
    "N_Days": 999,
    "Age": 21532,
    "Bilirubin": 2.3,
    "Cholesterol": 316.0,
    "Albumin": 3.35,
    "Copper": 172.0,
    "Alk_Phos": 1601.0,
    "SGOT": 179.8,
    "Tryglicerides": 63.0,
    "Platelets": 394.0,
    "Prothrombin": 9.7,
    "Stage": 3.0
}


# Send a POST request to the prediction endpoint and get the response
response = requests.post(url, json=customer).json()

# Print the response
print(response)

# Interpret the response and print the Cirrhosis status
if response['status'] == 'C':
    print('Your Cirrhosis status is C.')
elif response['status'] == 'CL':
    print('Your Cirrhosis status is CL.')
elif response['status'] == 'D':
    print('Your Cirrhosis status is D.')
