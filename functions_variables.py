import pandas as pd

def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    # Ensure tags column exists
    if 'tags' not in df:
        print("Warning: 'tags' column not found in DataFrame.")
        return df  # Return unchanged if no tags column

    # Ensure tags is a list
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    # Reset index to avoid indexing issues
    df = df.reset_index(drop=True)

    # Explode the tags column to create a new row for each tag
    df_exploded = df.explode('tags')

    # Apply One-Hot Encoding using get_dummies()
    df_tags_encoded = pd.get_dummies(df_exploded['tags'], prefix='tag')

    # Group by index and apply max() to keep 1s
    df_tags_encoded = df_tags_encoded.groupby(df_exploded.index).max()

    # Merge back with the original DataFrame
    df = df.drop(columns=['tags']).merge(df_tags_encoded, left_index=True, right_index=True)

    return df

def encode_primary_photo(df):
    """
    Converts the 'primary_photo' column into a binary (True/False) column 
    based on whether an href exists.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'primary_photo'.

    Returns:
        pandas.DataFrame: Modified DataFrame with 'has_primary_photo' column.
    """
    if 'primary_photo' not in df:
        print("Warning: 'primary_photo' column not found in DataFrame.")
        return df  # Return unchanged if the column is missing

    # Create a new binary column: True if 'href' exists, False otherwise
    df['has_primary_photo'] = df['primary_photo'].apply(lambda x: isinstance(x, dict) and 'href' in x)

    # Drop the original 'primary_photo' column
    df = df.drop(columns=['primary_photo'])

    return df