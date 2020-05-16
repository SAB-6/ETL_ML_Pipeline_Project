# import libraries
import sys
import re
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sqlalchemy import create_engine
import pickle
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from string import punctuation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load clean data from database and returns the
     feature (X) and target (Y) variables
     - Split data into features and target variables
     - Splits data to training and test sets
     - Tokenises and normalises the text data
     - Feature extraction using CountVectorizer and Tfidf
     - Create a pipepline 
    
    Inputs:
        database_filepath- database file path
    
    Returns: 
        X(feature), Y(targets)
        
    """
    
    # load data from database
    engine = create_engine("sqlite:///%s"%database_filepath) 
    df = pd.read_sql('data_cleaned', engine)
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis = 1).values
    
    # column names for the target columns
    category_names = list(df.drop(['id','message','original','genre'], axis=1).columns)
   
    return X, Y, category_names

def tokenize(text):
    """
    This function performs the underlisted process
        on each list of text
        - Removes punctuations
        - Remove stopwords,
        - Normalises the texts
        - Removes whitespaces        
    
    Params:
        text: list of text to be tokenized
    
    Returns:
        clean copy of tokenized text
    """
    # Remove punctuations  
    text = text.translate(str.maketrans('', '', punctuation))
    
    
    # tokenize text
    tokens = word_tokenize(text)
    
    #instantiantiate lemmmantizer
    lemmatizer = WordNetLemmatizer()

    #lower token, strip whitespaces and remove stopwords
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens


def build_model():
    """ Build machine learning model through
    pipeline. The hyperparameters as been hand tunned
        
    Params:
        None
    
    Returns:
        model, and category names
        """
    #split data to training and test sets
    #X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42) 
    
    #construct a pipeline to perform all prepocessing tasks and model instantiation sequentially
    pipeline_rf = Pipeline([
    ('count', CountVectorizer(lowercase=False, tokenizer=tokenize, stop_words= 'english', ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    

    
      
    return pipeline_rf

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Model evaluation
    
    Params:
        model - fitted model
        X_test - test feature
        Y_test - test targets
        category_names = target column names
     
    Returns:
        None
     """
    Y_pred = model.predict(X_test)
    Y_pred_t = np.transpose(Y_pred)
    Y_test_t = np.transpose(Y_test)
    
    #print evaluation metrics suich as 
    #accuracy, precision, recall and f1-score
    i = 0
    while i < Y_pred.shape[1]:
        print('Accuracy:\n {}'.format((Y_pred_t[i]== Y_test_t[i]).mean()))
        print(category_names[i],':\n',classification_report(Y_test_t[i], Y_pred_t[i]))
        i+=1    
   
def save_model(model, model_filepath):
    """
        Save model to a directory
    Params:
        model: model object
        model_filepath: path to save the model into
    Returns:
        None
    """
    joblib.dump(model, model_filepath) 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()