from sklearn.feature_extraction import text
from collections import Counter
import numpy as np
import pandas as pd


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
        return 0
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
        return 0
    num_words = len(tokens)
    c = Counter(tokens)
    ttr = float(len(c))/float(num_words)
    ttr = ttr*10
    ttr -= 5
    if ttr < 0: ttr = 0
    return ttr


def main():
    df     = pd.read_csv("data/lyrical_genius.csv")

    df = df[((df["Genre"] == "pop") | (df["Genre"] ==  "country"))]
    df = df.drop(columns="Unnamed: 0")
    df = df.drop_duplicates(subset=["Name","Artist"],keep=False)

    df.head()

    corpus = df['Lyrics']
    corpus = corpus.fillna("")
    corpus = corpus.tolist()
    vectorizer = text.TfidfVectorizer(decode_error='ignore', 
            strip_accents='unicode', stop_words='english',
            max_features=10)
    vectorizer.fit(corpus)
    feature_names = vectorizer.get_feature_names()
    feature_names = feature_names + ["TTR", "Avg_Wrdlen"]
    print(feature_names)
    X = get_lyrics_features(corpus, vectorizer)
    for ind, feature in enumerate(feature_names):
        df[feature] = X[:,ind]
    print(df.tail)

if __name__ == '__main__':
    main()
