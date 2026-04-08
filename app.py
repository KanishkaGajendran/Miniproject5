import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

df=pd.read_excel("D:\Data _Science\Projects\Mini-Project 5\chatgpt_style_reviews_dataset.xlsx")
df['date'] = (df['date'].astype(str).replace('########', pd.NA)).astype('datetime64[ns]')
df['date'] = df['date'].ffill()

def rating_sentiment(rating):
  if rating >= 4:
    return "Positive"
  elif rating == 3:
    return "Neutral"
  else:
    return "Negative"

df["sentiment"] = df["rating"].apply(rating_sentiment)

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard – ChatGPT Reviews")

st.header("Dataset Overview")
st.dataframe(df)

st.header("1️⃣ Overall Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

st.header("2️⃣ Sentiment vs Rating")
st.write(pd.crosstab(df['rating'], df['sentiment']))

st.header("3️⃣ Keywords by Sentiment")
sentiment_option = st.selectbox("Select Sentiment", df['sentiment'].unique())
text = " ".join(df[df['sentiment'] == sentiment_option]['review'])

if text:
        wc = WordCloud(width=800, height=400).generate(text)
        plt.imshow(wc)
        plt.axis("off")
        st.pyplot(plt)

st.header("4️⃣ Sentiment Over Time")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.to_period('M').astype(str)
trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
st.line_chart(trend)

st.header("5️⃣ Verified vs Non-Verified Users")
st.write(pd.crosstab(df['verified_purchase'], df['sentiment']))

st.header("6️⃣ Review Length vs Sentiment")
st.write(df.groupby('sentiment')['review_length'].mean())

st.header("7️⃣ Location-wise Sentiment")
st.write(pd.crosstab(df['location'], df['sentiment']))

st.header("8️⃣ Platform Comparison")
st.write(pd.crosstab(df['platform'], df['sentiment']))

st.header("9️⃣ Version-wise Sentiment")
st.write(pd.crosstab(df['version'], df['sentiment']))

st.header("🔟 Common Negative Feedback Themes")
negative_text = " ".join(df[df['sentiment'] == 'Negative']['review'])
words = negative_text.split()
common_words = Counter(words).most_common(10)
st.write(common_words)
