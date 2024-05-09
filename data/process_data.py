# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """Loads data from two CSV files: messages and categories.

    Args:
        messages_filepath (str): The filepath to the messages CSV file.
        categories_filepath (str): The filepath to the categories CSV file.

    Returns:
        pandas.DataFrame: A merged dataframe containing messages and their corresponding categories.
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

    # Split 'categories' column into 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # Extract column names from the first row and remove the last 2 characters
    category_colnames = categories.iloc[0].str[:-2].tolist()

    # Rename columns of 'categories'
    categories.columns = category_colnames

    # Convert string values to integers and clip them to 1
    categories = categories.apply(lambda x: x.str[-1:].astype(int).clip(upper=1))

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates 
    # Check for the number of duplicates and print the count
    duplicate_count = df.duplicated().sum()
    print(f"Found {duplicate_count} duplicates, dropping them.")

    # drop duplicates
    df.drop_duplicates(inplace=True)
    print("Cleaning completed...")
    return df 

def save_data(df, database_filename):

    """Saves the given dataframe `df` to an SQLite database at the specified filepath `database_filename`.

    Args:
        df (pandas.DataFrame): The dataframe to be saved.
        database_filename (str): The filepath for the SQLite database.

    Returns:
        None
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("MessagesCategoriesTable", engine, index=False, if_exists='replace')  


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
                'and the file path for the database to save the cleaned data. '\
                'For example: python process_data.py messages.csv categories.csv '\
                'DisasterResponse.db')



if __name__ == '__main__':
    main()