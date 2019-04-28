from sklearn.feature_extraction import text
from collections import Counter
import numpy as np
import pandas as pd

seven = 7

def get_lyrics_features(corpus, vectorizer):
    """ inputs: 
            corpus - an iterable of strings
            vectorizer - a trained instance of text.TfidfVectorizer
        outputs: a matrix of features derived from the lyrics
    """
    n=len(corpus)
    d = len(vectorizer.get_feature_names())+2
    X = np.empty((n,d))
    for ind, lyrics in enumerate(corpus):
        if lyrics != "":
            tfidf = vectorizer.transform([lyrics])
            tfidf = tfidf.toarray()[0].tolist()
            cmplx = get_complexity(lyrics, vectorizer)
            row = tfidf + cmplx
            X[ind] = row
        else:
            X[ind] = np.zeros(d)
    return X

def get_complexity(lyrics, vectorizer):
    """ inputs:
            lyrics - a string consisting of the lyrics of one song
            vectorizer - a trained instance of text.TfidfVectorizer
        returns features related to the complexity of the lyrics
    """
    # convert lyrics to sequence of tokens
    tokenize = vectorizer.build_tokenizer()
    tokens = tokenize(lyrics)
    # token-type ratio: the ratio of unique words to total words
    ttr = get_ttr(tokens)
    avg_wrdlen = get_avg_wrdlen(tokens)
    return [ttr, avg_wrdlen]

def get_avg_wrdlen(tokens):
    """ given a list of tokens, derives the average wordlength
        adapted from code from NLP final project Fall2018
    """
    if len(tokens) < 2:
        return -1
    num = len(tokens)
    count = 0
    for word in tokens:
        count += len(word)
    avg_wrdlen = float(count)/float(num)
    avg_wrdlen = avg_wrdlen
    if avg_wrdlen < 0: avg_wrdlen = 0
    return avg_wrdlen

def get_ttr(tokens):
    """ given a list of tokens, derives a metric of the ratio of 
        unique words to total words.
        adapted from code from NLP final project Fall2018
    """
    if len(tokens) < 2:
        return -1
    num_words = len(tokens)
    c = Counter(tokens)
    ttr = float(len(c))/float(num_words)
    ttr = ttr
    if ttr < 0: ttr = 0
    return ttr


def fit_transform_lyrics(df, max_features, max_df=0.5, min_df=0.1):
    """ inputs: df - the initial dataframe. Must have a "Lyrics" col
            max_features, max_df, min_df - the equivalent params for
                scikit learn's text.TfidfVectorizer()
                By default, max_df=0.5, min_df=0.1
        outputs: df - the dataframe with max_features+2 new columns
            encoding information about the lyrics
            vectorizer: the fitted text.TfidfVectorizer()
    """
    corpus = df['Lyrics']
    corpus = corpus.fillna("")
    corpus = corpus.tolist()
    vectorizer = text.TfidfVectorizer(decode_error='ignore', 
            strip_accents='unicode', stop_words='english',
            max_features=max_features,
            max_df=max_df, min_df=min_df)
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names()
    feature_names = feature_names + ["TTR", "Avg_Wrdlen"]
    print(feature_names)
    X = get_lyrics_features(corpus, vectorizer)
    for ind, feature in enumerate(feature_names):
        df[feature] = X[:,ind]
    return df, vectorizer

def genre_transform_lyrics(df, max_features):
    """ inputs: df - the initial dataframe. Must have a "Lyrics" col
            max_features, max_df, min_df - the equivalent params for
                scikit learn's text.TfidfVectorizer()
                By default, max_df=0.5, min_df=0.1
        outputs: df - the dataframe with max_features+2 new columns
            encoding information about the lyrics
            vectorizer: the fitted text.TfidfVectorizer()
        transforms lyrics in df into one document per genre before
        vectorization
    """
    lyric_list = df['Lyrics']
    lyric_list = lyric_list.fillna("")
    lyric_list = lyric_list.tolist()
    labels = df['Genre']
    genres = labels.unique().tolist()
    labels = labels.tolist()
    lyrics_by_label = {genre:"" for genre in genres}
    for i in range(len(lyric_list)):
        lyrics_by_label[labels[i]] += " " + lyric_list[i]

    corpus = [lyrics_by_label[key] for key in lyrics_by_label]
    
    vectorizer = text.TfidfVectorizer(decode_error='ignore', 
            strip_accents='unicode', stop_words='english',
            max_features=max_features)
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names()
    feature_names = feature_names + ["TTR", "Avg_Wrdlen"]
    print(feature_names)
    X = get_lyrics_features(lyric_list, vectorizer)
    for ind, feature in enumerate(feature_names):
        df[feature] = X[:,ind]
    return df, vectorizer

