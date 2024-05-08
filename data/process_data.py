# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge two CSV datasets: messages and categories.

    Parameters:
    - messages_filepath (str): Filepath to the messages dataset.
    - categories_filepath (str): Filepath to the categories dataset.

    Returns:
    pandas.DataFrame: Merged DataFrame containing messages and their corresponding categories.
    """
    print("Loading datasets...")
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories =  pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')
    print("Loading Completed and df created...")

    return df

def clean_data(df):
    """
    Clean the input DataFrame by splitting the 'categories' column,
    extract column names and converting to list, renaming columns, 
    converting category values to binary, dropping 
    duplicates, concatenate original df with the new 'categories',
    and saving the cleaned data to a CSV file.

    Parameters:
    - df: Input DataFrame containing a 'categories' column.

    Returns:
    df: Cleaned DataFrame with individual category columns.
    """
    print("Cleaning df...")

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Using a lambda function to extract column names and converting to list
    category_colnames = list(row.apply(lambda x: x[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]# set each value to be the last character of the string
        categories[column] = categories[column].astype(int)# convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: 0 if (x == 0) else 1) #convert integers larger than zero to 1

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates 
    print("Droping {} duplicates.".format(df.duplicated().sum()))

    # drop duplicates
    df.drop_duplicates(inplace=True)
    print("Cleaning completed...")
    return df 

def save_data(df, database_filename):
    """Function saves given dataframe df to SQL database with filepath database_filename
    
    Parameters:
    - df: dataframe to be saved
    - database_filename: SQL database filepath
    
    Return:
    None - dataframe is saved."""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("Table", engine, index=False, if_exists='replace')  


def main():
        if len(sys.argv) == 4:

            messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

            df = load_data(messages_filepath, categories_filepath)

            print('Cleaning data...')
            df = clean_data(df)
            
            print('Saving data...\n    DATABASE: {}'.format(database_filepath))
            save_data(df, database_filepath)
            
            print('Cleaned data saved to database!')
        
        else:
            print('Please provide the file paths for the messages and categories, '\
                'as well as the file path for the database to save the cleaned data. '\
                'For example: python process_data.py messages.csv categories.csv '\
                'DisasterResponse.db')



if __name__ == '__main__':
    main()