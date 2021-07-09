import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

#nltk initializers
from nltk.sentiment import SentimentIntensityAnalyzer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import pandas as pd
data = pd.read_csv('LazadaReviews_en.csv',error_bad_lines=False,encoding = 'latin-1')
data = data.dropna(subset=['reviews'])
data['reviews'] = data['reviews'].str.replace(r'[^\w\s]+', '')
data_text = data[['reviews']]
data_text['index'] = data_text.index
documents = data_text

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
processed_docs = documents['reviews'].map(preprocess)
#bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
#     print(k, v)
    count += 1
    if count > 10:
        break


#bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
        
from gensim import models
new_model = gensim.models.ldamodel.LdaModel.load('lda_model.model')

output_scores = []
output_topics = []
sia = SentimentIntensityAnalyzer()


#Streamlit section
my_page = st.sidebar.radio('Sprint Navigation', ['Introduction', 'Data','Machine Learning','Demo','Contributors'])

if my_page == 'Introduction':
    st.title("Introduction")
    
    st.header("Problem Statement")
#     st.image(banner)
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    st.header("Objectives")
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    st.header("Use - Cases")
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
elif my_page == 'Data':
    st.title("The Dataset")
    
    st.header("Sources")
    
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    st.header("Web Scraping")
    
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    
    st.header("Data Cleaning")
    
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)

    
elif my_page == 'Machine Learning':
    st.title("Topic Modeling")
    
    st.header("Bag of Words")
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    
    st.header("IDF")
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
    st.header("LDA")
    st.markdown('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque varius dolor vel dolor tempus, vel gravida lorem finibus. Integer placerat placerat dolor, non lacinia lectus. Ut et dolor ultrices, viverra nibh non, laoreet leo. Ut sed mi in risus aliquam accumsan sit amet at nisl. Etiam tempus sapien ante, mattis rhoncus elit elementum vitae. Morbi eget lacus nec nibh ultrices varius eget sit amet nibh. Donec aliquam nunc sit amet sagittis consequat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam purus tellus, bibendum et condimentum pellentesque, efficitur non nisl. Curabitur cursus maximus est ut malesuada. Vestibulum ultricies sollicitudin dapibus. Morbi a sollicitudin elit, eu varius ligula. Pellentesque at semper nisl, at placerat lacus.',unsafe_allow_html=False)
    
#     st.image(knn1)
    
    
#     st.image(knn2)

elif my_page == 'Demo':
    st.title("Review Analyzer")
    st.markdown('This short demo allows us to analyze a new review and see which topic it is mostly associated with',unsafe_allow_html=False)
    st.markdown('sample text 1: Received my order in good condition. It was highly secured and  taken care off by the courier.  A very good item to haved.More goods to buy and purchase this seller, good quality for its price defitnely will order again',unsafe_allow_html=False)
    st.markdown('sample text 2: product is good but delivery is slow, I hate you because the product is excellent quality. I am impatient because the delivery is moderate',unsafe_allow_html=False)
    
    user_input = st.text_input("submit your review here")
    st.markdown('YOUR REVIEW')
    st.markdown(user_input)
    
    positive = sia.polarity_scores(user_input)['pos']
    negative = sia.polarity_scores(user_input)['neg']
    if positive > negative:
        st.header('positive review')
    elif positive < negative:
        st.header('negative review')
    else:
        st.header('neutral review')
       
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_vector = dictionary.doc2bow(preprocess(user_input))

    for index, score in sorted(new_model[bow_vector], key=lambda tup: -1*tup[1]):
        output_scores.append(score)
        output_topics.append(new_model.print_topic(index, 5))
    
    
    st.write(output_scores[0],'highest probability of review being part of these topics')
    
    s = output_topics[0].split("+",5)
    number_list = []
    number = 0
    while number < len(s):
        number_pro = float(s[number].split('*',1)[0])
        number_list.append(number_pro)
        number = number + 1
    #calculations to enlarge pie
    summer = sum(number_list)
    missing = 1 - summer
    portion = missing / len(number_list)
    new_portions = [x+portion for x in number_list]
    multiplied_portions = [x*100 for x in new_portions]
    # get word list
    word_list = []
    word = 0
    while word < len(s):
        word_pro = s[word].split('*',1)[1]
        word_pro = word_pro.replace('"','') #and word_pro.replace(" ","")
        word_list.append(word_pro)
        word = word + 1
    
    st.write('therefore, your review has the highest probability of being part of the topic (',word_list[0],') with the weight of:',multiplied_portions[0])
    
    from matplotlib import pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    ax.pie(multiplied_portions, labels = word_list,autopct='%1.2f%%')
    st.pyplot(fig)
    
elif my_page == 'Contributors':

    st.header('The Team')
    
    st.markdown('We are proud to present our research and are avid fans of Davide Guetta. We wish for David Guetta, good health and good beats.',unsafe_allow_html=False)

    col1, col2, col3 = st.beta_columns(3)
    
    with col1:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="fidel-ivan-racines-187477167" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/fidel-ivan-racines-187477167?trk=profile-badge">Fidel Ivan Racines</a></div>'''
                              ,height=350)
    
    with col2:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="jon-marco-francisco-77592214a" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/jon-marco-francisco-77592214a?trk=profile-badge%22%3EJon Marco Francisco</a></div>'''
                              ,height=350)
    
    with col3:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="ajloconer" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/ajloconer?trk=profile-badge%22%3EAndrew Justin Oconer</a></div>'''
                              ,height=350)

    col4, col5, col6 = st.beta_columns(3)

    with col4:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="preciouseunicegrullo" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/preciouseunicegrullo?trk=profile-badge%22%3EPrecious Eunice Grullo</a></div>'''
                              ,height=350)
        
    with col5:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="adrianjanairo" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/adrianjanairo?trk=profile-badge%22%3EAdrian Genevie Janairo</a></div>'''
                              ,height=350)
        
    with col6:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="HORIZONTAL" data-vanity="reiniel-dan-pablo-5197641b4" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/reiniel-dan-pablo-5197641b4?trk=profile-badge%22%3EReiniel Dan Pablo</a></div>
                              ''', height=350)
        
    st.header('The Mentor')
    
    st.markdown('Our mentor has helped and guided us through out the process of this study.',unsafe_allow_html=False)
    
    col7, col8, col9 = st.beta_columns(3)
    
    with col7:
        st.components.v1.html('''''', height=350)
        
    with col8:
        st.components.v1.html('''<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                              <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="patricknuguid" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://ph.linkedin.com/in/patricknuguid?trk=profile-badge%22%3EPatrick Nuguid</a></div>'''
                              ,height=350)
        
    with col9:
        st.components.v1.html('''''', height=350)
        
    st.header('The Organization')
    
#     st.image(banner2)
    
    st.markdown('Eskwelabs is an online data upskilling school for people and teams in Southeast Asia. Who gives access opportunities in the future of work through accessible data skills that are high in-demand as the amount of data in the world increases exponentially.',unsafe_allow_html=False)
    
    st.markdown('Our mission is to give access to engaging and future-relevant skills education is then crucial to help people and teams thrive in that future. In Southeast Asia, where more than half of the population is under the age of 30, we believe data education can democratize access to meaningful careers for workers and sustainable competitiveness for companies.',unsafe_allow_html=False)
    
    st.markdown('At the same time, learning happens in all kinds of ways. Many learning environments, both in school and online, rely on lecture formats which are rarely engaging and effective for technical skills. Eskwelabs aims to enable participatory and active learning experiences so beyond acquiring in-demand skills, we can also rediscover the joy of learning and reinventing ourselves.',unsafe_allow_html=False)
