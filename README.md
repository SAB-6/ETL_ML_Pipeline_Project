# Disaster Response ETL/ML Pipeline Project
<img scr ="https://miyamotointernational.com/wp-content/uploads/disaster-response.jpg"/>

ETL and Machine learning pipeline using python
The ETL/ML project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. 

The Project is subdivided into:
1. Preprocessing of data and ETL Pipeline (including data storage into a sql database)
2. Building trained model through machine Learning Pipeline and GridSearchCV to tune the model hyperparameters
3. Model deployment(Web app) with flask app.

# Dependencies
- Python 3+
- NumPy==1.12.1
- Pandas==2.0.15
- Sciki-Learn 0.23+
- NLTK (Natural Language Process Libraries)3.2.5+
- SQLAlchemy 2.3+
- Flask 0.12+
- Plotly==2.0.15
- gunicorn 19.9+

# Instruction on how to run the program while in the main project directory (ETL_ML_Pipleine_project)
## - To run the ETL pipeline (which extracts the data, transforms it and loads it into the database) type:
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
## - To run Machine learning (ML) pipeline(which perform feature extractio, trains, predict and saves the model type:
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
## - To run the app:
        Navigate to the app directory and type python run.py
        Then go to http://0.0.0.0:3001/

