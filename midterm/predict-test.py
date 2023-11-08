import requests

url = 'http://0.0.0.0:4041/predict'

person = {
    "age" : 28, 
    "workclass:": "Private", 
    "final_weight": 338409 , 
    "education": "Bachelors", 
    "educationnum": 13,
    "marital_status":"Married-civ-spouse", 
    "occupation": "Prof-specialty", 
    "relationship": "Wife" , 
    "race": "Black", 
    "gender": "Female",
    "capital_gain": 0, 
    "capital_loss":0, 
    "hours_per_week":40, 
    "native_country": "cuba"
}
try:
    response = requests.post(url, json=person)
    
    if response.status_code == " <=50K":
        data = response.json()
        if data.get('status') == 1:
            print('Earns less than 50k')
        else:
            print('Earns more than 50k.')
    else:
        print(f'Request failed with status code {response.status_code}')
except requests.exceptions.RequestException as e:
    print(f'Request error: {e}')