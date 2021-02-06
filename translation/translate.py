import pandas as pd
import os
import re

from google.cloud import translate

def prepare_data():
    # read data
    train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=200000)
    valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv")
    test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv")
    
    # select random 40% train samples out of all data
    train_df = (
        train_df[train_df.similarity != "-"]
        .sample(frac=0.4, random_state=42)
        .reset_index(drop=True)
    )
    train_df.dropna(axis=0, inplace=True)
    valid_df = (
        valid_df[valid_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    valid_df.dropna(axis=0, inplace=True)
    test_df = (
        test_df[test_df.similarity != "-"]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    test_df.dropna(axis=0, inplace=True)
    
    return train_df, valid_df, test_df
        
def process_dataset(df, path):
    # create copy to modify
    df = df.copy()
    print(f"Dataframe size {df.shape[0]}")
    # create new df
    new_df = pd.DataFrame(columns=['similarity', 'sentence1', 'sentence2'])
    batch_size = 128
    k = 0
    l = len(df.index)
    batch_counter = 1
    
    while k < l:
        rows = []
        batch = []
        indeces = df.index[k:k+batch_size]
        for row_i in indeces:
            try:
                row = df.iloc[row_i]
            except:
                continue
            if not row['sentence1'].endswith('.'):
                row['sentence1'] += '.'
            rows.append([row['sentence1'], row['sentence2'], row['similarity']])
            batch.append(row['sentence1'] + " <SEP> " + row['sentence2'])
            
        if not len(batch):
            print(f"Batch {batch_counter} was translated")
            batch_counter += 1
            k += batch_size
            continue

        translated_batch = translate_text(batch)
        # create pairs
        for i, row in enumerate(rows):
            sentence1 = row[0]
            sentence2 = row[1]
            similarity = row[2]
            
            try:      
                translations = translated_batch[i].split(" <SEP> ")

                translated_sentence1 = translations[0]
                translated_sentence2 = translations[1]
            except:
                continue
            # append original text
            new_df = new_df.append(dict({'sentence1': sentence1, 'sentence2': sentence2, 'similarity': similarity}), ignore_index=True)
            # append translation
            new_df = new_df.append(dict({'sentence1': sentence1, 'sentence2': translated_sentence2, 'similarity': similarity}), ignore_index=True)
            new_df = new_df.append(dict({'sentence1': translated_sentence1, 'sentence2': sentence2, 'similarity': similarity}), ignore_index=True)
            new_df = new_df.append(dict({'sentence1': translated_sentence1, 'sentence2': translated_sentence2, 'similarity': similarity}), ignore_index=True)
            
        print(f"Batch {batch_counter} was translated")
        batch_counter += 1
        
        k += batch_size

    print("Dataframe was translated")
    
    new_df.to_csv(path)
    print(f"Dataframe was saved to {path}")
    

def translate_text(texts, target="uk", project_id="engaged-kite-304010"):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    
    response = client.translate_text(
        request = {
            "parent": parent,
            "contents": texts,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": target,
        }
    )
    
    return [translation.translated_text for translation in response.translations]


def process_file(file):
    return process_text(file.read())

def process_text(text):
    text = text if type(text) == str else text.decode("utf-8") 

    # put text in all lower case letters 
    text = text.lower()
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove html special characters (like &nbsp)
    text = re.sub(r'&\w+;\s*', '', text)
    # remove html tags
    text = re.sub(r'<[^>]*>', '', text)
    # remove digits
    text = re.sub(r'\d*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)

    # remove all non-alphanumeric chars
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub("  ", " ", text)
    text = re.sub("   ", " ", text)

    return text

def create_text_column(df, file_directory='plagiarism_corpus/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`. 
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''
   
    # create copy to modify
    text_df = df.copy()
    
    # store processed text
    text = []
    
    # for each file (row) in the df, read in the file 
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)
    
    # add column to the copied dataframe
    text_df['Text'] = text
    
    return text_df

def create_text_pairs(df):
    '''Convert df to the format `text_a`, `text_b`, `target`'''
    new_df = pd.DataFrame()

    text_a = []
    text_b = []
    targets = []

    for task, task_df in df.groupby(['Task']):
        orig_text = task_df[task_df['Category']==-1]['Text'].item()

        for index, row in task_df.iterrows():
            text_a.append(orig_text)
            text_b.append(row['Text'])
            targets.append(row['Target'] if row['Target'] != -1 else 1)

    new_df['Text A'] = text_a
    new_df['Text B'] = text_b
    new_df['Target'] = targets

    return new_df

def prepare_plagiarism_data(file_directory='plagiarism_corpus/'):
    csv_file = file_directory+ 'file_information.csv'
    plagiarism_df = pd.read_csv(csv_file)
    
    # convert Category column to numerical values
    cat_num = {'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1}
    plagiarism_df['Category'] = plagiarism_df['Category'].map(cat_num)
    
    # add Target column
    plagiarism_df['Target'] = [ x if x < 1 else 1 for x in plagiarism_df['Category'] ]
    
    # read files
    plagiarism_df = create_text_column(plagiarism_df)
    # create text pairs from files
    plagiarism_df = create_text_pairs(plagiarism_df)
    
    return plagiarism_df


def process_plagiarism_data(df, path='plagiarism_data/corpus.csv'):
    # create copy to modify
    df = df.copy()
    print(f"Dataframe size {df.shape[0]}")
    # create new df
    new_df = pd.DataFrame(columns=['Text A', 'Text B', 'Target'])
    
    for row_i in df.index:
        row = df.iloc[row_i]
        
        text_a = row['Text A']
        text_b = row['Text B']
        target = row['Target']

        translations = translate_text([text_a, text_b])

        translated_text_a = translations[0]
        translated_text_b = translations[1]
        # append original text
        new_df = new_df.append(dict({'Text A': text_a, 'Text B': text_b, 'Target': target}), ignore_index=True)
        # append translation
        new_df = new_df.append(dict({'Text A': text_a, 'Text B': translated_text_b, 'Target': target}), ignore_index=True)
        new_df = new_df.append(dict({'Text A': translated_text_a, 'Text B': text_b, 'Target': target}), ignore_index=True)
        new_df = new_df.append(dict({'Text A': translated_text_a, 'Text B': translated_text_b, 'Target': target}), ignore_index=True)

    print("Dataframe was translated")
    
    new_df.to_csv(path)
    print(f"Dataframe was saved to {path}")
    

if __name__ == "__main__":
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcloud-auth.json"
    
    # Translate SNLI corpus
    train_df, valid_df, test_df = prepare_data()
    print("Dataset was prepared")
    
    process_dataset(train_df, 'data/train_test1.csv')
    process_dataset(valid_df, 'data/valid_test1.csv')
    process_dataset(test_df, 'data/test_test1.csv')
    
    # Translate plagiarism corpus
    plagiarism_df = prepare_plagiarism_data()
    process_plagiarism_data(plagiarism_df)
    