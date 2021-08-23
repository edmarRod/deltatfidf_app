from deltatfidf import DeltaTfidfVectorizer
import pandas as pd


def get_vocab_idf(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    dtf = DeltaTfidfVectorizer(**kwargs)
    dtf.fit(df['text'], df['label'])

    vocab = pd.DataFrame(zip(dtf.idf_, dtf.get_feature_names()), columns=['Delta-Idf', 'Word'])

    return vocab.sort_values(['Delta-Idf']).reset_index(drop=True)
