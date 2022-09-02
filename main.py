import streamlit as st
st.set_page_config(page_title= 'Team A UN SDG Classifier')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="white", palette=None, rc = custom_params)
import plotly.express as px
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize 
import re
import pdfplumber
import pickle
from textwrap import wrap

filename = 'final_model.sav'
model = pickle.load(open(filename, 'rb'))

####################### for background image ###################################
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpg')

tabs_font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 26px;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)
#################### functions required ########################################
from nltk.corpus import stopwords 
stopwords= stopwords.words('english')
def cleanText(text):
    text = re.sub(r'''!\(\)-\[]\{};:'"\,<>./?@#$%^&*_~''', r' ', text) 
    text = text.lower()
    text = text.replace(',', '')
    tokens = nltk.word_tokenize(text)
    wordlist = [] 
    for w in tokens:
      if w not in stopwords:
        if w.isalpha():
          wordlist.append(w)

    clean_text = ' '.join(wordlist)
    return clean_text

#
sdg_names = {
        1: 'No Poverty', 2: 'Zero Hunger', 3: 'Good Health and Well-being', 4: 'Quality Education',
        5: 'Gender Equality', 6: 'Clean Water and Sanitation', 7: 'Affordable and Clean Energy',
        8: 'Decent Work and Economic Growth',9: 'Industry, Innovation and Infrastructure', 10: 'Reduced Inequality',
        11: 'Sustainable Cities and Communities', 12: 'Responsible Consumption and Production', 13: 'Climate Action',
        14: 'Life Below Water', 15: 'Life on Land', 16: 'Peace and Justice Strong Institutions', 17: 'Partnerships to achieve the Goal'
    }

######################## Frontend code  ##############################################
body = st.container()

def print_imgs(li):
    n= 1
    for i in li:
        if n==6:
            n= 1
        if n==1:
            c1.image('imgs/s'+str(i)+'.jpg')
            c1.markdown('Probability : '+str(vals[n-1]))
        elif n==2:
            c2.image('imgs/s'+str(i)+'.jpg')
            c2.markdown('Probability : '+str(vals[n-1]))
        elif n==3:
            c3.image('imgs/s'+str(i)+'.jpg')
            c3.markdown('Probability : '+str(vals[n-1]))
        elif n==4:
            c4.image('imgs/s'+str(i)+'.jpg')
            c4.markdown('Probability : '+str(vals[n-1]))
        elif n==5:
            c5.image('imgs/s'+str(i)+'.jpg')
            c5.markdown('Probability : '+str(vals[n-1]))
        n += 1

def predict(text):
    #clean_text= cleanText(text)
    text = [text]
    prediction = model.predict_proba(text)
    prediction = prediction[0]
    sdgs = []
    for index,value in enumerate(prediction):
        if value > 0.09:
            sdgs.append(index+1)
            vals.append(round(value, 2))
    return [prediction , sdgs]

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_text())
        text = ' '.join(data)
    return text # build more code to return a dataframe 

##################################### Functions for report #################################

def plot_predictors(sample):
    sample = sample.split(' ')
    vectoriser = model['vectoriser']
    selector = model['selector']
    clf = model['clf']
    top_n = 15
    features =  model['vectoriser'].get_feature_names_out()
    if selector is not None:
        features = features[selector.get_support()]
    axis_names = [f'feature_{x + 1}' for x in range(top_n)]

    if len(clf.classes_) > 2:
        results = list()
        for c, coefs in zip(clf.classes_, clf.coef_):
            idx = coefs.argsort()[::-1][:top_n]
            results.extend(tuple(zip([c] * top_n, features[idx], coefs[idx])))
    else:
        coefs = clf.coef_.flatten()
        idx = coefs.argsort()[::-1][:top_n]
        results = tuple(zip([clf.classes_[1]] * top_n, features[idx], coefs[idx]))

    df_lambda = pd.DataFrame(results, columns =  ['sdg', 'feature', 'coef'])

    predictors = []
    for i, word in enumerate(df_lambda['feature'].values):
        if word in sample:
            predictors.append((df_lambda['sdg'][i], word, df_lambda['coef'][i]))

    if len(predictors): 
        df_pred = pd.DataFrame(predictors, columns = ['sdg', 'predictors', 'coef'])
        df_pred.sort_values(['sdg', 'coef'], ignore_index = True, inplace = True)
        colors = px.colors.qualitative.Dark24[:15]
        template = 'SDG: %{customdata}<br>Predictor: %{y}<br>Coefficient: %{x:.2f}'

        fig = px.bar(
            data_frame = df_pred,
            x = 'coef',
            y = 'predictors',
            custom_data = ['sdg'],
            facet_col = 'sdg',
            facet_col_wrap = 3,
            facet_col_spacing = .15,
            orientation = 'h',
            height = 600,
            labels = {
                'coef': 'Coefficient',
                'predictors': ''
            },
            #title = 'Key Predictor words in given document'
        )
        fig.update_traces(width= 0.5)
        fig.for_each_trace(lambda x: x.update(hovertemplate = template))
        fig.for_each_trace(lambda x: x.update(marker_color = colors.pop(0)))
        fig.for_each_annotation(lambda x: x.update(text = 'SDG : '+'<br>'.join(wrap(sdg_names[int(x.text.split("=")[-1])], 30))))
        fig.update_yaxes(matches = None, showticklabels = True)

        st.plotly_chart(fig, use_container_width= True)
    else:
        st.text('No predictors!')

def plot_prob(pred):
    fig, ax= plt.subplots()
    ax = sns.barplot(x = list(range(1, 16)), y = pred, color = '#86dc3d')
    plt.xlabel('Sustainable Development Goals')
    plt.ylabel('Prediction Probability')
    col1.pyplot(fig)

def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")

def create_wordcloud(text):
    wordcloud= WordCloud(background_color ='white', width= 800, height= 400).generate(text)
    wordcloud.recolor(color_func = black_color_func)
    fig, ax = plt.subplots() 
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

def word_freq(text):
    text_list = text.split(' ')
    fdist= FreqDist(text_list)
    flist = fdist.most_common(20)
    fdf = pd.DataFrame(flist, columns= ['Frequent Word', 'frequency'])
    fig, ax = plt.subplots()
    ax.barh(fdf['Frequent Word'], fdf['frequency'] , color = '#86dc3d')
    plt.title('Most frequent words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    col2.pyplot(fig)

#################################### MAIN ########################################
with body:
    new_title = '<p style="font-family:sans-serif; color:Black; font-size: 42px; text-align: center;">UN-SDGs Label Classifier Web App</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.image('imgs/intropm1_2.png')
    # explore,teamA,sdg=st.columns(3)
    # explore.image('imgs/SDG-logo-removebg.png')
    # teamA.image('imgs/teamA.jpg')
    # sdg.image('imgs/SDG_cubes1-removebg.png')

    tab1, tab2, tab3,tab4 = st.tabs(["📝Type or Paste","📑Sample text"," 📄Upload Text","🗒️Upload PDF"])

    with tab1:
        input_text = st.text_area('Type or paste a sample doucment here.')

    with tab2: 
        sample = st.selectbox('Or select sample text', ('Team_A.txt', None), index= 1)
        if sample != None:
            f = open("Team_A.txt", "r",encoding='utf-8')
            input_text = f.read()


    with tab3:
        txt_col=st.header('upload text file here')
        uploaded_file = txt_col.file_uploader('', type="txt")
        if uploaded_file is not None:
            input_text = str(uploaded_file.read(), 'utf-8')

    with tab4:
        pdf_col=st.header('upload PDF file here')
        uploaded_file = pdf_col.file_uploader('', type="pdf")
        if uploaded_file is not None:
            input_text = extract_data(uploaded_file)

    #Predicting
    vals = []
    input_text = cleanText(input_text)
    prediction , nums = predict(input_text)

    if  st.button('Get SDG labels'):
        if input_text == None or input_text == "":
            st.text('Please upload or choose document. \nPossibly uploaded document is empty.')
        else:   
            c1, c2, c3, c4, c5 = st.columns(5)        
            print_imgs(nums)
            with st.spinner('Generating report...'):
                #Freqdist and probability of sdg
                col1, col2 = st.columns(2)
                col1.subheader('Probability of each SDG.')
                plot_prob(prediction)
                col2.subheader('Frequent words')
                word_freq(input_text)

                #key predictors
                st.subheader('Key predictor words in given document')
                plot_predictors(input_text)

                #word cloud
                st.subheader('Word Cloud')
                create_wordcloud(input_text)
