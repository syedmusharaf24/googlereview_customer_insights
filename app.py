import streamlit as st
from google.oauth2 import service_account

# Ensure you have your secrets configured in .streamlit/secrets.toml
# as per the previous instructions for Streamlit Secrets.
credentials = service_account.Credentials.from_service_account_info(st.secrets["google"])

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
# These downloads are only attempted if the data is not found,
# which is good practice for Streamlit Cloud deployments.
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

def scrape_Maps_reviews(url):
    driver = None # Initialize driver to None for proper cleanup in case of early errors
    try:
        # Resolve short URLs if needed
        if 'goo.gl' in url or 'maps.app.goo.gl' in url:
            url = resolve_short_url(url)
            st.write(f"Resolved URL: {url}")

        # Setup Chrome options for headless execution on Streamlit Cloud
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # IMPORTANT FOR STREAMLIT CLOUD: Specify the paths to Chromium and ChromeDriver
        # These paths are where `chromium-browser` and `chromium-chromedriver`
        # are installed by the `packages.txt` file.
        chrome_options.binary_location = "/usr/bin/chromium-browser"
        service = webdriver.ChromeService(executable_path="/usr/bin/chromium-chromedriver")

        # Initialize the driver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)

        # Wait for the page to load initial content
        time.sleep(5)

        # Try to find and click the "Reviews" tab with multiple possible selectors
        # This makes the scraper more robust to different language settings or minor UI changes.
        review_selectors = [
            "button[aria-label*='Reviews']",
            "button[aria-label*='Reseñas']",  # Spanish
            "button[aria-label*='Avis']",     # French
            "button[aria-label*='Bewertungen']",  # German
            "button[aria-label*='レビュー']",  # Japanese
            "button[aria-label*='评论']"      # Chinese
        ]

        clicked = False
        for selector in review_selectors:
            try:
                # Wait until the reviews tab is clickable and then click it
                reviews_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", reviews_tab)
                clicked = True
                time.sleep(3) # Give time for the reviews section to load
                break
            except:
                continue

        if not clicked:
            st.warning("Could not find reviews tab. Trying to proceed anyway...")

        # Scroll to load more reviews
        scroll_pause_time = 2
        scroll_count = 20 # Number of times to scroll down

        # Find the scrollable reviews section. Google Maps review sections can have different selectors.
        scrollable_section = None
        try:
            scrollable_section = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
        except:
            try:
                # Another common selector for the scrollable reviews container
                scrollable_section = driver.find_element(By.CSS_SELECTOR, "div[aria-label*='Reviews for']")
            except:
                st.warning("Could not find scrollable reviews section")

        if scrollable_section:
            for _ in range(scroll_count):
                # Scroll down within the specific scrollable element
                driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_section)
                time.sleep(scroll_pause_time)
        else:
            # Fallback to window scrolling if a specific scrollable section isn't found
            st.warning("Falling back to window scrolling for reviews.")
            for _ in range(scroll_count):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)

        # Extract review elements using multiple robust selectors
        review_elements = []
        review_selectors = [
            "div.jftiEf",                  # Primary current selector for individual review blocks
            "div.gws-localreviews__google-review", # Older or alternative selector
            "div.review-item",             # Generic fallback
            "div[data-review-id]"          # Another potential unique identifier
        ]

        for selector in review_selectors:
            review_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if review_elements:
                st.info(f"Found {len(review_elements)} review elements using selector: {selector}")
                break
        
        if not review_elements:
            st.warning("No review elements found with any known selectors after scrolling. Dataframe might be empty.")

        reviews = []
        for review in review_elements:
            try:
                # Extract reviewer name
                reviewer_name = "Unknown"
                try:
                    reviewer_name = review.find_element(By.CSS_SELECTOR, "div.d4r55").text
                except:
                    try:
                        reviewer_name = review.find_element(By.CSS_SELECTOR, "div.TSUbDb").text
                    except:
                        pass # Keep "Unknown" if no specific element found

                # Extract rating (out of 5 stars)
                rating = None
                try:
                    # Look for the span with aria-label like "4.5 stars"
                    rating_element = review.find_element(By.CSS_SELECTOR, "span.kvMYJc[aria-label*='stars']")
                    aria_label = rating_element.get_attribute("aria-label")
                    if aria_label:
                        # Extract the numeric part from "X.X stars"
                        rating_match = re.search(r'(\d+\.?\d*)\s*stars', aria_label)
                        if rating_match:
                            rating = float(rating_match.group(1))
                except:
                    # Fallback for older structures or different rating representations
                    try:
                        rating_element = review.find_element(By.CSS_SELECTOR, "span.viS8Zb")
                        rating = float(rating_element.get_attribute("aria-label").split()[0])
                    except:
                        pass


                # Extract review text
                review_text = ""
                try:
                    # Try to click "More" button if present to expand full text
                    more_button = review.find_element(By.CSS_SELECTOR, "button.jEseJe")
                    driver.execute_script("arguments[0].click();", more_button)
                    time.sleep(0.5) # Small delay for text to expand
                except:
                    pass # No "More" button, continue to extract available text

                try:
                    review_text = review.find_element(By.CSS_SELECTOR, "span.wiI7pd").text
                except:
                    try:
                        review_text = review.find_element(By.CSS_SELECTOR, "div.Jtu6Td").text
                    except:
                        pass # Keep empty string if no review text found


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
                st.warning(f"Error processing an individual review element: {str(e)}")
                continue

        driver.quit() # Always quit the driver

        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred during scraping: {str(e)}")
        if driver: # Ensure driver is initialized before attempting to quit
            driver.quit()
        return None

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions (@...) and numbers
        text = re.sub(r'\@\w+|\#|\d+', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text.strip() # Remove leading/trailing whitespace
    return ""

# Function to analyze sentiment
def analyze_sentiment(text):
    if not text: # Handle empty text
        return 'Neutral'
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
    if not text.strip(): # Check if there's actual text
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Streamlit app
def main():
    st.set_page_config(page_title="Customer Review Insights", layout="wide")
    st.title("Customer Review Insights :bar_chart:")
    st.write("Analyze customer reviews from Google Maps locations. Enter a Google Maps URL below to get started.")

    # Input for Google Maps link
    gmaps_link = st.text_input("Enter Google Maps URL of the place:", help="e.g., https://maps.app.goo.gl/YourPlaceID")

    if st.button("Fetch Reviews"):
        if gmaps_link:
            with st.spinner('Fetching reviews from Google Maps... This may take a moment, especially for many reviews.'):
                df = scrape_Maps_reviews(gmaps_link)

            if df is not None and not df.empty:
                st.success(f"Successfully retrieved {len(df)} reviews!")
                process_data(df)
            else:
                st.error("Failed to retrieve reviews or no reviews found. Please double-check the URL and ensure it's a public Google Maps listing with reviews.")
        else:
            st.warning("Please enter a Google Maps URL to fetch reviews.")

def process_data(df):
    st.markdown("---")
    st.header("Review Data Overview")

    # Display the raw data
    st.subheader("Raw Review Data (First 5 Rows)")
    st.dataframe(df.head())

    # Data Cleaning and Sentiment Analysis
    with st.spinner("Performing sentiment analysis and data cleaning..."):
        df['Cleaned_Review'] = df['Review'].apply(clean_text)
        df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)
    st.success("Data processing complete!")

    st.markdown("---")
    st.header("Sentiment Analysis Results")

    # Display sentiment distribution
    st.subheader("Distribution of Review Sentiments")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment',
                             title='Distribution of Review Sentiments',
                             color='Sentiment',
                             color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                             hole=0.3)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Display rating distribution if available
    if 'Rating' in df.columns and df['Rating'].notna().any():
        st.subheader("Distribution of Ratings")
        fig_rating = px.histogram(df, x='Rating', nbins=5, title='Distribution of Ratings',
                                  category_orders={"Rating": [1, 2, 3, 4, 5]},
                                  color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_rating.update_layout(bargap=0.1)
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info("Rating information not available or not properly extracted for analysis.")

    st.markdown("---")
    st.header("Word Cloud Analysis")

    col1_wc, col2_wc = st.columns(2)

    # Word Cloud for All Reviews
    with col1_wc:
        st.subheader("Overall Review Word Cloud")
        all_reviews = " ".join(df['Cleaned_Review'].tolist())
        if all_reviews.strip():
            wordcloud_fig = generate_wordcloud(all_reviews)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.write("Not enough text to generate an overall word cloud.")
        else:
            st.write("No cleaned review text available for overall word cloud.")

    # Common words in positive and negative reviews
    with col2_wc:
        st.subheader("Common Words by Sentiment")
        
        positive_reviews = " ".join(df[df['Sentiment'] == 'Positive']['Cleaned_Review'].tolist())
        negative_reviews = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Review'].tolist())

        pos_col, neg_col = st.columns(2)

        with pos_col:
            st.markdown("##### Positive Reviews")
            if positive_reviews.strip():
                pos_wordcloud_fig = generate_wordcloud(positive_reviews)
                if pos_wordcloud_fig:
                    st.pyplot(pos_wordcloud_fig)
                else:
                    st.write("Not enough positive reviews for a word cloud.")
            else:
                st.write("No positive reviews to generate a word cloud.")

        with neg_col:
            st.markdown("##### Negative Reviews")
            if negative_reviews.strip():
                neg_wordcloud_fig = generate_wordcloud(negative_reviews)
                if neg_wordcloud_fig:
                    st.pyplot(neg_wordcloud_fig)
                else:
                    st.write("Not enough negative reviews for a word cloud.")
            else:
                st.write("No negative reviews to generate a word cloud.")

    st.markdown("---")
    st.header("Temporal Analysis (if available)")

    # Sentiment by day of week if date is available
    if 'Date' in df.columns and df['Date'].notna().any():
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed

            if not df.empty:
                df['Day_of_Week'] = df['Date'].dt.day_name()
                # Group by day of week and calculate average sentiment
                df['Sentiment_Score'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
                sentiment_by_day = df.groupby('Day_of_Week')['Sentiment_Score'].mean().reset_index()

                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                sentiment_by_day['Day_of_Week'] = pd.Categorical(sentiment_by_day['Day_of_Week'], categories=days_order, ordered=True)
                sentiment_by_day = sentiment_by_day.sort_values('Day_of_Week')

                st.subheader("Average Sentiment by Day of Week")
                fig_day_sentiment = px.bar(sentiment_by_day, x='Day_of_Week', y='Sentiment_Score',
                                             title='Average Sentiment Score by Day of Week',
                                             color='Sentiment_Score',
                                             color_continuous_scale=['red', 'gray', 'green'])
                st.plotly_chart(fig_day_sentiment, use_container_width=True)
            else:
                st.info("No valid date data found after processing for temporal analysis.")
        except Exception as e:
            st.warning(f"Could not analyze sentiment by day due to an issue with date column: {str(e)}")
            st.info("Ensure the 'Date' column in the scraped data is in a recognizable date format.")
    else:
        st.info("Date information not available or not properly extracted for temporal analysis.")


    st.markdown("---")
    st.subheader("Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name="processed_reviews.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()