# DisasterResponseProject
Machine learning project utilizing disaster Response datasets

# Project Motivation
Welcome to the Disaster Response ML Project! This project focuses on implementing an ETL process and a Machine Learning Pipeline to classify tweets and messages. The goal is to aid in disaster emergency response efforts through a Flask Web App.

# Instructions
This project is structured into three main sections:

- Data Processing: Includes building an ETL pipeline to extract data from the source, clean it, and store it in a SQLite database.
- Machine Learning Pipeline: Develops a pipeline capable of classifying tweet text messages into 36 different categories.
- Web Application: Running a web application that displays model results in real-time.

# Installations 
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Processing Libraries: NLTK
- SQLite Database Libraries: SQLAlchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

  
# File Descriptions
### app
- **template**
  - **master.html**: Main page of the web app.
  - **go.html**: Classification result page of the web app.
- **run.py**: Flask file that runs the web app.

### data
- **disaster_categories.csv**: Data to process.
- **disaster_messages.csv**: Data to process.
- **process_data.py**: Script for processing data.
- **InsertDatabaseName.db**: Database to save clean data to.

### models
- **train_classifier.py**: Script for training the classifier.
- **classifier.pkl**: Saved model.

# How to Interact with the project
To clone the git repository, use the following command:
- git clone https://github.com/abg3/Disaster-Response-ML-Project.git

Running the Application
Navigate to the project's directory and execute the following commands to set up the database, train the model, and save it:

To run the ETL pipeline for cleaning data and storing it in the database:
- python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

 To run the ML pipeline that loads data from the database, trains the classifier, and saves it as a pickle file: 

-  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Finally, run the following command in the app's directory to start the web app:

- python run.py

# Licensing, Authors, Acknowledgements.
Special thanks to Udacity and Figure Eight for providing the dataset and an opportunity to work on this project.
