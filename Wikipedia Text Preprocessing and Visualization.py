# Apply text preprocessing and visualization to the dataset containing Wikipedia texts.



import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import Word, TextBlob
from warnings import filterwarnings

filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

df = pd.read_csv("C:\\Users\\hazal\\OneDrive\\Masaüstü\\datasets\\wiki_data.csv")
df.head()

# Step 1

#  Create a function named clean_text for text preprocessing. The function should perform the following tasks:
# • Convert text to lowercase,
# • Remove punctuation,
# • Remove numeric expressions.

def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace("[^\w\s]","")
    text = text.str.replace("\n","")
    # Numbers
    text = text.str.replace("\d","")
    return text

df["text"] = clean_text(df["text"])

# step 2
# Create a function named `remove_stopwords` that will remove
# unimportant words during the feature extraction process.
import nltk
nltk.download("stopwords")
def remove_stopwords(text):
    stop_words = stopwords.words("english")
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])

# step 3
# Find the words with low frequency in the text.
pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]

# step 4
# Remove the low-frequency words from the text.
sil = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"]= df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Step 5 5 (Tokenize the texts)
from textblob import Word, TextBlob
df["text"].apply(lambda x: TextBlob(x).words)


# Step 6
# Lemmatization
import pandas as pd
from textblob import Word
df["text"] = df["text"].apply(lambda x:" ".join([Word(word).lemmatize() for word in x.split()]))

# Step 7
# Frequency calculation

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# Step 8
# Create a barplot graph.

# Naming the columns
tf.columns = ["words","tf"]
# Visualize the words that appear more than 2000 times.
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Visualizing words with WordCloud.
text = " ".join(i for i in df["text"])

# Let's specify the features for visualizing the word cloud.
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Step 9
# Text preprocessing
# Add visualization operations as arguments to the function.
# Write a 'docstring' explaining the function.

df = pd.read_csv("C:\\Users\\hazal\\OneDrive\\Masaüstü\\datasets\\wiki_data.csv")

def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemleri yapar
    :param text: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: text

    example:
            wiki_preprocess(dataframe[col_name])
    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace("[^\w\s]","")
    text = text.str.replace("\n","")
    # numbers
    text = text.str.replace("\d","")
    # stopwords
    sw = stopwords.words("english")
    text = text.apply(lambda x: "".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(" ".join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    if Barplot:
        # Calculating Term Frequencies
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Column naming
        tf.columns = ["words","tf"]
        # Visualization of words occurring more than 2000 times
        tf[tf["tf"] > 2000].plot.bar(x="words",y="tf")
        plt.show()

    if Wordcloud:
        # We combined the words
        text = " ".join(i for i in text)
        # We define the properties for the wordcloud visualization.
        wordcloud = Wordcloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.show()
    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)
