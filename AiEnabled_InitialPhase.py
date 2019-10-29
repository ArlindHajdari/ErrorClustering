import h2o
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import cntk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from collections import Counter
import json
import random
from TopToolbar import TopToolbar

data_set = []
train = pd.read_csv("incidents.csv")

Des = train.iloc[:, 2].values
# ID = train.iloc[:, 16].values
# Sev = train.iloc[:, 3].values
 
# load nltk's English stopwords as variable called 'stopwords'
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


def get_jaccard_sim(list1, list2):
    a = set(list1)
    b = set(list2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def tokenize_stem_stopword_puncfree(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stemmed = " ".join(stemmer.stem(word) for word in normalized.split())
    processed = re.sub(r"\d+", "", stemmed)
    return processed.split()


# Tokenize and stem
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# Tokenize the sentences
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# Accepts list of lists(with words), and tries to classify by similarity
def cluster_by_similarity(dataset, similarity_percentage):
    final_dictionary = {}
    similarity_percentage /= 100
    for sentence in dataset:
        for key, value in final_dictionary.items():
            if get_jaccard_sim(sentence.split(), key.split()) > similarity_percentage:
                final_dictionary[key].append(sentence)
                break
        else:
            final_dictionary[sentence] = [sentence.split()]

    return final_dictionary


# Apply cleaning and stemming
for i in Des:
#     y = tokenize_and_stem(i)
    data_set.append(i)


# Clustering by similarity
classified_by_similarity = cluster_by_similarity(data_set, 80)
classified_by_similarity_no_stemmed = [error for errors in list(classified_by_similarity) for error in tokenize_only(errors)]
classified_by_similarity_stemmed = [error for errors in list(classified_by_similarity) for error in tokenize_and_stem(errors)]

vocab_frame = pd.DataFrame({"words": classified_by_similarity_no_stemmed}, index=classified_by_similarity_stemmed)

# Save transformed data to json file
# with open("clustered_by_similarity.json", "w+") as fileWrite:
#     json.dump(classified_by_similarity.keys(), fileWrite, sort_keys=True, indent=4, separators=(',', ': '))


# Read the data from the file that we previously wrote on!
# with open("TestingAlg.json", "r") as clusteredwords_file:
#     loaded_json = json.load(clusteredwords_file)
#     clusteredwords_fromjson = [item.split(",") for item in dict(loaded_json).keys()]


# Tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=int(len(classified_by_similarity)/2),
                                   min_df=0.1, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(list(classified_by_similarity))
terms = tfidf_vectorizer.get_feature_names()
# Cosine similarity of the synopses
dist = 1 - cosine_similarity(tfidf_matrix)


# K-Means model creation
# km = KMeans(n_clusters=len(terms))
#
# km.fit(tfidf_matrix)
# joblib.dump(km,  'error_cluster.pkl')


# K-means model loading
km = joblib.load('error_cluster.pkl')
clusters = km.labels_.tolist()


# Multidimensional scaling
MDS()
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)

pos = mds.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]

cluster_colors = {index: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for index in range(0, len(terms))}
cluster_names = {index: terms[index] for index in range(0, len(terms))}

plot_data = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=list(classified_by_similarity)))

# Group by cluster
groups = plot_data.groupby('label')


# Matplot method of plotting, using errors description as title(hard readable)
# ==================================================================================================================================
# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelleft=False)

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(plot_data)):
    ax.text(plot_data.ix[i]['x'], plot_data.ix[i]['y'], plot_data.ix[i]['title'], size=8)

plt.show()  # show the plot

# ==================================================================================================================================


# Html plotting
# ==================================================================================================================================
# define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
# Optional, just adds 5% padding to the autoscaling
ax.margins(0.03)

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                     label=cluster_names[name], mec='none',
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in terms]

    # set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                             voffset=10, hoffset=10, css=css)
    # connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    # set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
# show legend with only one dot
ax.legend(numpoints=1)
# =============================================================================================================

# Works only on python versions 2.6-2.7 and 3.2-3.4
# mpld3.display()

# Generate html and save file
# html = mpld3.fig_to_html(fig)
# with open("error_clustering.html", "w+") as error_html:
#     error_html.write(html)
