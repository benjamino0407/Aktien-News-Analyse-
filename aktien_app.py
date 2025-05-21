import streamlit as st
from transformers import pipeline
from newsapi import NewsApiClient

# Titel der App
st.title("📈 Aktien-News Analyse")

# Eingabefeld für Aktien/Schlüsselwörter
keyword = st.text_input("🔍 Gib ein Stichwort ein (z. B. Amazon, Bitcoin, Rheinmetall):", "Amazon")

# NewsAPI-Key (kostenloser Account bei newsapi.org nötig)
newsapi = NewsApiClient(api_key='dein_newsapi_key_hier')

# Lade aktuelle News
all_articles = newsapi.get_everything(q=keyword, language='de', page_size=5)

# Lade das Sentiment-Analyse-Modell
sentiment = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

# Zeige die Ergebnisse
for article in all_articles['articles']:
    st.write("📰 **Titel:**", article['title'])
    result = sentiment(article['title'])[0]
    st.write("📊 **Stimmung:**", result['label'], f"({result['score']:.2f})")
    st.markdown("---") 
