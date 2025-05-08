import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from urllib.parse import urlparse

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def resolve_short_url(short_url):
    """Resolve short URLs to their final destination"""
    try:
        session = requests.Session()
        resp = session.head(short_url, allow_redirects=True)
        return resp.url
    except Exception as e:
        st.warning(f"Could not resolve short URL: {e}")
        return short_url

def scrape_google_maps_reviews(url):
    try:
        # Resolve short URLs if needed
        if 'goo.gl' in url or 'maps.app.goo.gl' in url:
            url = resolve_short_url(url)
            st.write(f"Resolved URL: {url}")

        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(5)  # Increased wait time
        
                # Try to find and click the "Reviews" tab with multiple possible selectors
        review_selectors = [
            "button[aria-label*='Reviews']",  # English
            "button[aria-label*='Reseñas']",  # Spanish
            "button[aria-label*='Avis']",     # French
            "button[aria-label*='Bewertungen']",  # German
            "button[aria-label*='レビュー']",  # Japanese
            "button[aria-label*='评论']"      # Chinese
        ]
        
        clicked = False
        for selector in review_selectors:
            try:
                reviews_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", reviews_tab)
                clicked = True
                time.sleep(3)
                break
            except:
                continue
        
        if not clicked:
            st.warning("Could not find reviews tab. Trying to proceed anyway...")
        
        # Scroll to load more reviews
        scroll_pause_time = 2
        scroll_count = 20
        
        # Find the scrollable reviews section
        scrollable_section = None
        try:
            scrollable_section = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
        except:
            try:
                scrollable_section = driver.find_element(By.CSS_SELECTOR, "div.m6QErb[aria-label*='Reviews']")
            except:
                st.warning("Could not find scrollable reviews section")
        
        if scrollable_section:
            for _ in range(scroll_count):
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_section)
                time.sleep(scroll_pause_time)
        else:
            # Fallback to window scrolling
            for _ in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
        
        # Extract review elements with more robust selectors
        review_elements = []
        review_selectors = [
            "div.jftiEf",  # Primary selector
            "div.gws-localreviews__google-review",  # Alternative selector
            "div.review-item"  # Another alternative
        ]
        
        for selector in review_selectors:
            review_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if review_elements:
                break
        
        reviews = []
        for review in review_elements:
            try:
                # Extract reviewer name
                reviewer_name = ""
                try:
                    reviewer_name = review.find_element(By.CSS_SELECTOR, "div.d4r55").text
                except:
                    try:
                        reviewer_name = review.find_element(By.CSS_SELECTOR, "div.TSUbDb").text
                    except:
                        reviewer_name = "Unknown"
                
                # Extract rating (out of 5 stars)
                rating = None
                try:
                    rating_element = review.find_element(By.CSS_SELECTOR, "span.kvMYJc")
                    rating_style = rating_element.get_attribute("style")
                    rating_match = re.search(r'width:\s*(\d+)px', rating_style)
                    rating = int(rating_match.group(1)) / 14 if rating_match else None
                except:
                    try:
                        rating_element = review.find_element(By.CSS_SELECTOR, "span.viS8Zb")
                        rating = float(rating_element.get_attribute("aria-label").split()[0])
                    except:
                        pass
                
                # Extract review text
                review_text = ""
                try:
                    review_text = review.find_element(By.CSS_SELECTOR, "span.wiI7pd").text
                except:
                    try:
                        review_text = review.find_element(By.CSS_SELECTOR, "div.Jtu6Td").text
                    except:
                        pass
                
                # Extract date
                review_date = ""
                try:
                    review_date = review.find_element(By.CSS_SELECTOR, "span.rsqaWe").text
                except:
                    try:
                        review_date = review.find_element(By.CSS_SELECTOR, "span.dehysf").text
                    except:
                        pass
                
                if review_text:  # Only add if we got at least the review text
                    reviews.append({
                        'Reviewer': reviewer_name,
                        'Rating': rating,
                        'Review': review_text,
                        'Date': review_date
                    })
            except Exception as e:
                st.warning(f"Error processing a review: {str(e)}")
                continue
        
        driver.quit()
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        return df
    
    except Exception as e:
        st.error(f"Error scraping reviews: {str(e)}")
        try:
            driver.quit()
        except:
            pass
        return None

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'\@\w+|\#|\d+', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Streamlit app
def main():
    st.title("Customer Review Insights")
    st.write("Analyze customer reviews from Google Maps locations")
    
    # Input for Google Maps link
    gmaps_link = st.text_input("Enter Google Maps URL of the place:")
    
    if gmaps_link:
        with st.spinner('Fetching reviews from Google Maps...'):
            df = scrape_google_maps_reviews(gmaps_link)
            
        if df is not None and not df.empty:
            st.success(f"Successfully retrieved {len(df)} reviews!")
            process_data(df)
        else:
            st.error("Failed to retrieve reviews. Please check the URL and try again.")

def process_data(df):
    # Display the raw data
    st.subheader("Raw Data")
    st.dataframe(df.head())
    
    # Clean the reviews
    df['Cleaned_Review'] = df['Review'].apply(clean_text)
    
    # Analyze sentiment
    df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)
    
    # Display sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                 title='Distribution of Review Sentiments',
                 color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    st.plotly_chart(fig)
    
    # Display rating distribution if available
    if 'Rating' in df.columns:
        st.subheader("Rating Distribution")
        fig = px.histogram(df, x='Rating', nbins=5, title='Distribution of Ratings')
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
    
    # Word Cloud
    st.subheader("Word Cloud from Reviews")
    all_reviews = " ".join(df['Cleaned_Review'].tolist())
    if all_reviews.strip():  # Check if there's any text to generate wordcloud
        wordcloud_fig = generate_wordcloud(all_reviews)
        st.pyplot(wordcloud_fig)
    else:
        st.write("Not enough text to generate a word cloud.")
    
    # Sentiment by day of week if date is available
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Day_of_Week'] = df['Date'].dt.day_name()
            
            # Group by day of week and calculate average sentiment
            df['Sentiment_Score'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
            sentiment_by_day = df.groupby('Day_of_Week')['Sentiment_Score'].mean().reset_index()
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sentiment_by_day['Day_of_Week'] = pd.Categorical(sentiment_by_day['Day_of_Week'], categories=days_order, ordered=True)
            sentiment_by_day = sentiment_by_day.sort_values('Day_of_Week')
            
            st.subheader("Average Sentiment by Day of Week")
            fig = px.bar(sentiment_by_day, x='Day_of_Week', y='Sentiment_Score', 
                         title='Average Sentiment Score by Day of Week',
                         color='Sentiment_Score',
                         color_continuous_scale=['red', 'gray', 'green'])
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Could not analyze sentiment by day: {str(e)}")
    
    # Common words in positive and negative reviews
    st.subheader("Common Words by Sentiment")
    
    positive_reviews = " ".join(df[df['Sentiment'] == 'Positive']['Cleaned_Review'].tolist())
    negative_reviews = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Review'].tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Positive Reviews Word Cloud")
        if positive_reviews.strip():
            pos_wordcloud_fig = generate_wordcloud(positive_reviews)
            st.pyplot(pos_wordcloud_fig)
        else:
            st.write("Not enough positive reviews for a word cloud.")
    
    with col2:
        st.write("Negative Reviews Word Cloud")
        if negative_reviews.strip():
            neg_wordcloud_fig = generate_wordcloud(negative_reviews)
            st.pyplot(neg_wordcloud_fig)
        else:
            st.write("Not enough negative reviews for a word cloud.")
    
    # Download the processed data
    st.subheader("Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="processed_reviews.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()