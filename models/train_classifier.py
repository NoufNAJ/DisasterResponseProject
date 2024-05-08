# import libraries
import sys
import pandas as pd

#sqlalchemy database:
from sqlalchemy import create_engine
#regex:
import re
#nltk text processing libraries:
import nltk
nltk.download('omw-1.4')
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#pipline 
from sklearn.pipeline import Pipeline
#estimators:
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
#for training
from sklearn.model_selection import train_test_split
# for testing
from sklearn.metrics import classification_report
# for GridSearchCv
from sklearn.model_selection import GridSearchCV

#for saving model
import pickle


def load_data(database_filepath):
    """ Function loads data from a SQL-database at the specified path
    
    Parameters:
    SQL Database filepath
    
    Returns:
    X: features from database table
    y: labels from database table
    category_names: categories of labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Table", engine)

    # Define feature and target variables X and Y
    X = df['message'] # Feature
    Y = df[df.columns[4:]] # Target variable
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    
    # Remove URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
     # Lemmatize and lowercase tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

#model definition
def build_model():
    """
    Build a machine learning pipeline for multi-output classification using 
    CountVectorizer, TfidfTransformer, and RandomForestClassifier.
    
    Parameters:
    None

    Returns:
    GridSearchCV: Grid search object for hyperparameter tuning.

    This function constructs a machine learning pipeline and optimizes it 
    using grid search.
    """

   # Define the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to word count vectors
        ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output random forest classifier
    ])
    
    # Define parameters grid
    # parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     'tfidf__use_idf': (True, False),  # use IDF or not
    #     'clf__estimator__n_estimators': [10, 50, 100],  # number of trees in the forest
    #     'clf__estimator__min_samples_split': [2, 3, 4]  # minimum number of samples required to split an internal node
    # }
    # cv =  GridSearchCV(pipeline, param_grid=parameters)
    parameters = {'vect__ngram_range': ((1, 1),(1,2)),
                'clf__estimator__n_estimators': [10, 50],
                'clf__estimator__min_samples_split': [2, 5]}

    cv =  GridSearchCV(pipeline, param_grid=parameters,  n_jobs=-1, cv=2, verbose = 3)

    return cv 

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the performance of the trained model by predicting Y_pred based on X_test.
    
    For each column in Y_test and the corresponding elements in Y_pred, 
    generate a classification report including precision, f1-score, recall, and support.
    
    Parameters: 
    - model: Trained machine learning model
    - X_test: Test set features
    - Y_test: Test set labels

    Returns: 
    None
    """
    Y_pred = model.predict(X_test)

    # Iterate through each output category
    for i, col in enumerate(Y_test.columns):
        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i]))
        print("\n")

    return None

def save_model(model, model_filepath):
    """
    Save the given machine learning model to the specified file path.
    
    Parameters:
    - model: Trained machine learning model
    - model_filepath: File path to save the model
    
    Returns:
    None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)
    
    return None

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  random_state=42)      

        print('Building model...')
        model = build_model()        
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the file path of the disaster messages database '\
              'and the file path of the pickle file to save the model.'\
              ' \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

