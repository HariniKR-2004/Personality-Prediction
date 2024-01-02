from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import networkx as nx
import seaborn as sns
import nltk
from scipy import stats
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import math
import textstat
import base64
from io import BytesIO
import pickle as pk



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

with open('model.pkl', 'rb') as f:
    lmodel = pk.load(f)



def preprocess_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    # Join the processed words back into a string
    processed_text = " ".join(words)
    return processed_text

@app.route('/predict', methods=['GET', 'POST'])
def predict_personality():
    if request.method == 'POST':
        data = request.get_json()
        input_text = data.get('text')
        preprocessed_text = preprocess_text(input_text)
        # Make predictions using the pre-fitted model
        predictions = lmodel.predict([preprocessed_text])
        dicts={0:"ENFJ",1:"ENFP",2:"ENTJ",3:"ENTP",4:"ESFJ",5:"ESFP",6:"ESTJ",7:"ESTP",8:"INFJ",9:"INFP",10:"INTJ",11:"INTP",12:"ISFJ",13:"ISFP",14:"ISTJ",15:"ISTP"}
        # Return the predicted result
        return jsonify({'predictedType': dicts[predictions[0]]})
    return render_template('predict.html')

@app.route('/visualization')
def visualization():
# Count the occurrences of each personality type
    data=pd.read_csv("mbti.csv")

    type_counts = data['type'].value_counts(normalize=True).reset_index()
    type_counts.columns = ['Personality Type', 'Percentage']

    # Create an interactive bar chart
    fig1 = px.bar(type_counts, x='Personality Type', y='Percentage', title='Personality Type Distribution')
    fig1.update_layout(title_text='Personality Type Distribution')
    # Show the interactive chart
    plot1_html = fig1.to_html(full_html=False)


    #Create an interactive pie chart
    # Count the occurrences of each personality type
    type_counts = data['type'].value_counts(normalize=True).reset_index()
    type_counts.columns = ['Personality Type', 'Percentage']
    # Group personality types with less than 2% occurrence into 'Others'
    threshold = 0.011
    type_counts.loc[type_counts['Percentage'] < threshold, 'Personality Type'] = 'Others'
    # Create an interactive pie chart
    fig2 = px.pie(type_counts, names='Personality Type', values='Percentage', title='Personality Type Distribution', hole=0.2)
    fig2.update_traces(textinfo='percent+label', pull=[0.1] + [0] * (len(type_counts) - 1))
    fig2.update_layout(title_text='Personality Type Distribution', showlegend=False)
    # Show the interactive chart
    plot2_html = fig2.to_html(full_html=False)


    # Calculate the average post length for each personality type
    data['post_length'] = data['posts'].apply(len)
    avg_post_length = data.groupby('type')['post_length'].mean().reset_index()
    # Create an interactive bar chart with uniform intermediate red color
    fig3 = px.bar(
        avg_post_length,
        x='type',
        y='post_length',
        title='Average Post Length by Personality Type',
        labels={'type': 'Personality Type', 'post_length': 'Average Post Length'},
        color_discrete_sequence=['rgb(0, 0, 225)'] * len(avg_post_length)  # Uniform intermediate red color
    )
    # Customize the layout
    fig3.update_layout(title_text='Average Post Length by Personality Type')
    # Show the interactive chart
    plot3_html = fig3.to_html(full_html=False)



    #Personality type relationship type network
    # Assuming you have loaded the data into a DataFrame named 'data'
    # Count the occurrences of each personality type
    type_counts = data['type'].value_counts().reset_index()
    type_counts.columns = ['Personality Type', 'Count']
    # Create a graph where nodes are personality types and edges connect similar types
    G = nx.Graph()
    # Assuming you have a function to determine cognitive function similarity (customize this)
    def cognitive_function_similarity(type1, type2):
        # Define the cognitive functions for each personality type
        functions = {
            'I': 'Introversion', 'E': 'Extraversion',
            'N': 'Intuition', 'S': 'Sensing',
            'F': 'Feeling', 'T': 'Thinking',
            'J': 'Judging', 'P': 'Perceiving'
        }
        # Extract cognitive functions for each type
        functions_type1 = [functions[letter] for letter in type1]
        functions_type2 = [functions[letter] for letter in type2]
        # Calculate the similarity as the number of common cognitive functions
        common_functions = set(functions_type1) & set(functions_type2)
        return len(common_functions)
    # Add nodes to the graph
    G.add_nodes_from(type_counts['Personality Type'])
    # Connect nodes based on cognitive function similarity
    types = type_counts['Personality Type'].tolist()
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            if cognitive_function_similarity(types[i], types[j]) >= 2:
                G.add_edge(types[i], types[j])
    # Generate node positions using spring layout
    pos = nx.spring_layout(G)
    # Extract x and y positions for Plotly scatter plot
    x_pos = [pos[node][0] for node in G.nodes]
    y_pos = [pos[node][1] for node in G.nodes]
    # Create a DataFrame for Plotly express
    pos_df = pd.DataFrame({'Personality Type': list(G.nodes), 'X': x_pos, 'Y': y_pos})
    # Create an interactive scatter plot
    fig4 = px.scatter(
        pos_df, x='X', y='Y', text='Personality Type',
        title='Personality Type Relationship Network'
    )
    plot4_html = fig4.to_html(full_html=False)



    #TREE PLOT
    # Assuming you have the 'type_counts' DataFrame from previous code
    fig5 = go.Figure(go.Treemap(
        labels=type_counts['Personality Type'],
        parents=[''] * len(type_counts),
        values=type_counts['Count'],
        texttemplate="%{label}<br>%{value}",
        textposition="middle center",
        branchvalues="total",
        marker_colors=px.colors.qualitative.Plotly,
    ))
    fig5.update_layout(title_text='Personality Type Distribution (Tree Plot)')
    plot5_html = fig5.to_html(full_html=False)

    
    #Parallel Coordinate Plot
    # Add a numerical label for each personality type
    data['type_label'] = pd.Categorical(data['type']).codes
    # Create the Parallel Coordinates Plot
    fig6 = px.parallel_coordinates(
        data,
        color='type_label',  # Use the numerical label for coloring
        color_continuous_scale=px.colors.diverging.Tealrose,
        labels={'type_label': 'Personality Type'},
        title='Parallel Coordinates Plot'
    )
    plot6_html = fig6.to_html(full_html=False)


    #3D Scatter Plot of Personality Traits
    # Map personality types to numerical values
    data['I_E'] = data['type'].apply(lambda x: x[0])
    data['N_S'] = data['type'].apply(lambda x: x[1])
    data['T_F'] = data['type'].apply(lambda x: x[2])
    data['J_P'] = data['type'].apply(lambda x: x[3])
    type_mapping = {type_: idx for idx, type_ in enumerate(data['type'].unique())}
    data['type_numerical'] = data['type'].map(type_mapping)
    fig7 = go.Figure(data=[go.Scatter3d(
        x=data['I_E'],
        y=data['N_S'],
        z=data['T_F'],
        mode='markers',
        marker=dict(
            size=12,
            color=data['type_numerical'],  # Use numerical values for coloring
            opacity=0.8,
            colorscale='Viridis'  # You can choose a colorscale
        ),
        text=data['type'],  # Display the personality type when hovering over a point
    )])
    fig7.update_layout(title='3D Scatter Plot of Personality Traits',
                    scene=dict(
                        xaxis=dict(title='I_E'),
                        yaxis=dict(title='N_S'),
                        zaxis=dict(title='T_F')
                    ))
    plot7_html = fig7.to_html(full_html=False)



    #stem plot
    # Count the occurrences of each personality type
    type_counts = dict(data['type'].value_counts())
    # Plotting
    plt.figure(figsize=(20,10))
    plt.hlines(y=type_counts.keys(), xmin=0, xmax=type_counts.values(), color='slateblue')
    plt.plot(type_counts.values(), type_counts.keys(), 'o', color='slateblue')
    plt.title('Personality Type', fontsize=27)
    plt.xlabel('Data Count', fontsize=19)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, None)
    plt.tick_params(left=False)
    sns.despine(left=True)
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    stem_plot = base64.b64encode(img_buffer.read()).decode()


    #joint plot
    # Create a copy of the data
    data_copy = data.copy()
    # Define the function to count the number of words in each post
    def var_row(row):
        word_lengths = [len(post.split()) for post in row.split('|||')]
        return np.var(word_lengths)
    # Apply the function to calculate the variance of word counts and words per comment
    data_copy['variance_of_word_counts'] = data_copy['posts'].apply(lambda x: var_row(x))
    data_copy['words_per_comment'] = data_copy['posts'].apply(lambda x: len(x.split()) / 50)
    # Plotting
    plt.figure(figsize=(15, 10))
    sns.jointplot(x=data_copy["variance_of_word_counts"], y=data_copy["words_per_comment"], kind="hex")
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    joint_plot = base64.b64encode(img_buffer.read()).decode()



    #Swarm Plot of Word Count for Each Personality Type
    data = data.copy()
    # Count Number words for each post of a user
    def var_row(row):
        l = []
        for i in row.split('|||'):
            l.append(len(i.split()))
        return np.var(l)
    # Count Number words per post for total 50 posts in the whole row
    data['word_each_comment'] = data['posts'].apply(lambda x: len(x.split()) / 50)
    data['variance_word_count'] = data['posts'].apply(lambda x: var_row(x))
    plt.figure(figsize=(15, 10))
    sns.swarmplot(x="type", y="word_each_comment", hue="type", data=data, palette="viridis")
    plt.title('Swarm Plot of Word Count for Each Personality Type')
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    swarm_plot = base64.b64encode(img_buffer.read()).decode()


    #Outlier Analysis Distribution of post length
    # Create an interactive histogram
    fig8 = px.histogram(data, x='post_length', nbins=50, title='Distribution of Post Lengths')
    fig8.update_layout(
        title='Distribution of Post Lengths',
        xaxis_title='Post Length',
        yaxis_title='Frequency',
        bargap=0.05  # Adjust the gap between bars
    )
    plot8_html = fig8.to_html(full_html=False)


    #box plot for post length
    # Create an interactive boxplot
    fig9 = px.box(data, x='type', y='post_length', title='Post Length Distribution by Personality Type')
    fig9.update_layout(
        title='Post Length Distribution by Personality Type',
        xaxis_title='Personality Type',
        yaxis_title='Post Length',
        boxmode='group'  # Display boxplots for each category side by side
    )
    plot9_html = fig9.to_html(full_html=False)


    #Distribution of Post Lengths after removal of outliers
    # Calculate Z-scores
    z_scores = stats.zscore(data['post_length'])
    # Define a threshold for outliers (e.g., Z-score greater than 3 or less than -3)
    outlier_threshold = 3
    # Identify and remove outliers
    outliers = data[(z_scores > outlier_threshold) | (z_scores < -outlier_threshold)]
    data_no_outliers = data[(z_scores <= outlier_threshold) & (z_scores >= -outlier_threshold)]

    # Create an interactive histogram
    fig10 = px.histogram(data_no_outliers, x='post_length', nbins=50, title='Distribution of Post Lengths')
    fig10.update_layout(
        title='Distribution of Post Lengths after removal of outliers',
        xaxis_title='Post Length',
        yaxis_title='Frequency',
        bargap=0.05  # Adjust the gap between bars
    )
    plot10_html = fig10.to_html(full_html=False)

    #Feature Engineering
    nltk.download('punkt')
    nltk.download('stopwords')
    # Function for text processing (tokenization, stopword removal, and stemming)
    def process_text(text):
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Tokenization
        words = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        # Stemming
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]
        return " ".join(words)

    # Apply text processing to each text in the dataset
    data['processed_text'] = data['posts'].apply(process_text)
    # Create a bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['processed_text'])
    # Create a DataFrame with word frequencies
    word_frequencies = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # Sum the frequencies for each word across all texts
    total_word_frequencies = word_frequencies.sum()

    #Top 20 Most Frequent Words Across Personality Types
    # Display the most frequent words
    top_words = total_word_frequencies.sort_values(ascending=False).head(20)
    # Plot the word frequencies
    plt.figure(figsize=(12, 6))
    top_words.plot(kind='bar', color='skyblue')
    plt.title('Top 20 Most Frequent Words Across Personality Types')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    top20_freq_words = base64.b64encode(img_buffer.read()).decode()


    #sentimental Analysis
    # Function to calculate sentiment scores using TextBlob
    def calculate_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    # Apply the sentiment analysis function to each text in the dataset
    data['sentiment_score'] = data['processed_text'].apply(calculate_sentiment)
    # Function to categorize sentiment scores into positive, negative, or neutral
    def categorize_sentiment(score):
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'
    # Apply the sentiment categorization function to create a 'sentiment' column
    data['sentiment'] = data['sentiment_score'].apply(categorize_sentiment)


    #Average Sentiment Score by MBTI Type
    # Calculate average sentiment score for each MBTI type
    average_sentiment_by_type = data.groupby('type')['sentiment_score'].mean().sort_values()
    # Setting Seaborn style
    sns.set(style="whitegrid", palette="pastel")
    # Plotting the results
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=average_sentiment_by_type.index, y=average_sentiment_by_type.values, palette="viridis")
    ax.set_title('Average Sentiment Score by MBTI Type', fontsize=16)
    ax.set_xlabel('MBTI Type', fontsize=14)
    ax.set_ylabel('Average Sentiment Score', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    # Adding data labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                    textcoords='offset points')
    # Remove the top and right spines for aesthetics
    sns.despine()
    # Show the plot
    plt.tight_layout()
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    avg_sentiment_score = base64.b64encode(img_buffer.read()).decode()


    #Readability Score
    # Set a larger figure size
    plt.figure(figsize=(10, 6))
    data['readability'] = data['posts'].apply(lambda x: textstat.flesch_reading_ease(x))
    # Display readability distribution
    data['readability'].hist(bins=500, color='lightblue', edgecolor='black')
    # Set x-axis limits to focus on the range between 0 and 100
    plt.xlim(0, 100)
    # Add labels and title
    plt.xlabel('Readability Score')
    plt.ylabel('Frequency')
    plt.title('Readability Distribution')
    # Convert the plot to a base64-encoded image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    readability_score = base64.b64encode(img_buffer.read()).decode()

   
   #WORD CLOUD FOR THE WHOLE DATASET
    # Create a WordCloud for the entire dataset
    #html
    
    #Word Cloud for Each Personality type
    # Create WordClouds for each personality type arranged in a grid
    #html
    
    return render_template('visualization.html', plot1_html=plot1_html, plot2_html=plot2_html,plot3_html=plot3_html,plot4_html=plot4_html,plot5_html=plot5_html,plot6_html=plot6_html,plot7_html=plot7_html,stem_plot=stem_plot,joint_plot=joint_plot,swarm_plot=swarm_plot,plot8_html=plot8_html,plot9_html=plot9_html,plot10_html=plot10_html,top20_freq_words=top20_freq_words,avg_sentiment_score=avg_sentiment_score,readability_score=readability_score)



if __name__ == '__main__':
    app.run(debug=True)
