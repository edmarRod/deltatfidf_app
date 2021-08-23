import streamlit as st
import pandas as pd

from tfidf import get_vocab_idf

st.title('Sentiment Classification')

# Sidebar

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"],
                                             help="The labels must be 0 or 1 and the column names must be 'text' and 'label'",
                                             )

with st.sidebar.header("Display parameters"):
    idf_range = st.sidebar.slider(label="IDF Range", min_value=-7., max_value=7., step=.5, value=(-7., 7.))

with st.sidebar.header("Feature parameters"):
    df_range = st.sidebar.slider(label="Document frequency range", min_value=0.,
                                 max_value=1., step=.001, value=(0., 1.),
                                 help="Vocabulary outside this range will not be considered")

# Main page

st.subheader('1. Dataset')


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if st.checkbox(label="View dataset"):
        st.write(uploaded_file)

    vocab = get_vocab_idf(df, min_df=df_range[0], max_df=df_range[1])

    top_n = vocab.loc[vocab['Delta-Idf'].between(idf_range[0], idf_range[1])]\
        .sort_values('Delta-Idf', ascending=False).head(10)
    bottom_n = vocab.loc[vocab['Delta-Idf'].between(idf_range[0], idf_range[1])]\
        .sort_values('Delta-Idf', ascending=True).head(10)

    st.subheader("2. Most relevant words")
    right_col, left_col = st.columns(2)
    right_col.write("Top 10 most relevant words for majority class")
    right_col.dataframe(top_n)
    left_col.write("Top 10 most relevant words for minority class")
    left_col.dataframe(bottom_n)

    st.subheader("3. Word search")
    search_word = st.text_input("Input word to search:", )
    right_word, left_idf = st.columns(2)
    right_word.markdown("#### Word")
    left_idf.markdown("#### Delta-Idf")
    right_word.write(search_word)
    if vocab['Word'].isin([search_word]).any():
        found_idf = vocab.loc[vocab['Word'] == search_word, 'Delta-Idf'].values[0]
        left_idf.write(found_idf)
    else:
        left_idf.write("Word not found.")


else:
    st.write('Awaiting Dataset...')
