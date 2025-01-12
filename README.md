Name: Geetha N
Company:CODETECH IT SOLUTIONS 
ID:CT0806EK Domain: DATA SCIENCE
Duration:December 12th,2024 to january 12th,2025 
Mentor: Neha

OVERVIEW OF THE PROJECT

END-TO-END DATA SCIENCE PROJECT

This repository contains an end-to-end data science project aimed at predicting the survival of Titanic passengers based on their demographic and travel-related data. It demonstrates a complete data science workflow, including data preprocessing, model development, API deployment, and testing.

1. Problem Statement
The goal is to build a predictive model to determine whether a passenger survived the Titanic disaster based on features such as age, sex, ticket class, number of siblings/spouses aboard, number of parents/children aboard, and port of embarkation.

2. Project Workflow
The project follows a structured pipeline:

Data Collection & Understanding:

The Titanic dataset is used, containing information about passengers and their survival status.
Data Preprocessing:

Handle missing values (e.g., impute missing ages with the median).
Encode categorical variables like sex and embarked into numerical formats.
Drop irrelevant features such as deck, class, and who.
Model Development:

Train a machine learning model (Random Forest Classifier) to predict survival.
Evaluate the model using appropriate metrics such as accuracy, precision, recall, and F1-score.
Model Serialization:

Save the trained model and preprocessing encoders as serialized files for deployment.
API Development:

Develop a Flask-based API to serve predictions.
Endpoint /predict accepts passenger details in JSON format and returns the survival prediction.
Deployment:

Deploy the API locally for testing and usage.
3. Features
Data Preprocessing:
Cleaning and transforming raw data into a format suitable for machine learning models.
Machine Learning:
Building a classification model to predict survival.
API Integration:
Making predictions accessible via a Flask API endpoint.
Testing:
Validating the API using tools like Postman.
4. Tools and Technologies
Programming Language: Python
Libraries:
Data Processing: pandas, numpy
Machine Learning: scikit-learn
API Development: Flask
Version Control: Git
Testing Tools: Postman
5. Key Deliverables
Pickle Files:
titanic_model.pkl: Trained machine learning model.
sex_encoder.pkl and embarked_encoder.pkl: Encoders for categorical features.
Flask Application:
An API for predicting Titanic passenger survival.
Documentation:
Detailed project overview and instructions for running the application.
6. Instructions to Run the Project
Setup Environment:

Install required libraries from requirements.txt.
Train the Model:

Run the training script to preprocess data and train the model.
Start the API:

Launch the Flask application using python app.py.
Test the API:

Use Postman or any API client to test the /predict endpoint.
7. Learning Outcomes
Hands-on experience with data preprocessing and model training.
Exposure to API development for serving machine learning models.
Understanding the end-to-end lifecycle of a data science project.

![image](https://github.com/user-attachments/assets/442feaa2-4278-433e-a1d4-0cb4def950f8)

![image](https://github.com/user-attachments/assets/a44e2473-8320-490b-8e39-b07f650f239e)

![image](https://github.com/user-attachments/assets/ac477265-04e0-46be-8744-92b25167a2bf)

![image](https://github.com/user-attachments/assets/cfebb076-cf95-4c39-b724-2e9af5ffb242)






