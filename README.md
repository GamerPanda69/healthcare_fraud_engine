# healthcare_fraud_engine
-> create a virtual environment inside the folder healthcare_fraud_engine using command "python -m venv venv"(or python3 -m venv venv)
-> activate it using "source venv/bin/activate"
-> if (venv) is shown then you can start installing dependencies(requirements)
-> run command "pip install -r requirements.txt"
->login to kaggle and go to https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?select=Test_Outpatientdata-1542969243754.csv
-> download the dataset zip(8 csv files), create a folder called anomaly on the main(inside healthcare_fraud_engine) and extract those 8 files into it. 
->run the application with the command "python3 -m uvicorn main:app --reload"
