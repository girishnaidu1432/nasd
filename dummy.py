import streamlit as st
import datetime
import json
import openai
import re
from bs4 import BeautifulSoup
import requests
from langchain.document_loaders import WebBaseLoader

# OpenAI API Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# Bing Search URL
BING_SEARCH_URL = "https://www.bing.com/search?q="

def get_bing_results(query, num_results):
    """Fetch search results from Bing"""
    search_url = BING_SEARCH_URL + query.replace(" ", "+")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return [], "Failed to fetch results from Bing"
    
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    
    for b in soup.find_all('li', class_='b_algo')[:num_results]:
        title = b.find('h2').text if b.find('h2') else "No Title"
        link = b.find('a')['href'] if b.find('a') else "No Link"
        snippet = b.find('p').text if b.find('p') else "No snippet available"
        results.append({"title": title, "link": link, "snippet": snippet})
    
    return results, None

def scrape_full_page(url):
    """Scrape content from a webpage and extract stock ticker & price"""
    try:
        loader = WebBaseLoader(url)
        doc = loader.load()
        soup = BeautifulSoup(doc[0].page_content, "html.parser")
        full_content = soup.get_text(separator="\n")
        
        # Extract stock ticker symbol (assumes capital letters and 4-letter ticker)
        ticker_match = re.search(r'\b[A-Z]{2,5}\b', full_content)
        ticker = ticker_match.group(0) if ticker_match else "N/A"

        # Extract stock price (simple regex for price format detection)
        price_match = re.search(r'\$\d{1,5}(\.\d{1,2})?', full_content)
        price = price_match.group(0) if price_match else "N/A"

        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"url": url, "date": date, "full_content": full_content, "ticker": ticker, "price": price}
    
    except Exception as e:
        return {"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"}

def query_openai_llm(user_query, scraped_data):
    """Generate AI response based on scraped data"""
    prompt = f"""
    You are an expert financial analyst. Answer the user's question based on the scraped search data.
    Give responses in bullet points.

    User Query: {user_query}
    Scraped Data: {json.dumps(scraped_data, indent=2)}

    AI Response:
    """
    
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    
    return response['choices'][0]['message']['content'] if response else "No response generated."

def validate_results(scraped_data):
    """Validate stock data for correctness"""
    if not scraped_data:
        return "No data available for validation."

    stock_data = [
        {
            "url": item["url"],
            "date": item["date"],
            "ticker": item.get("ticker", "N/A"),
            "price": item.get("price", "N/A"),
            "timestamp": item["date"],
        }
        for item in scraped_data
    ]

    prompt = f"""
    You are an expert financial analyst. Validate the following stock price data.
    Ensure answers are based on Eastern Standard Time (EST) and follow strict validation rules.

    Data:
    {json.dumps(stock_data, indent=2)}

    Provide results in this table format:

    | Validation Criteria                | Result    | Reason                                  |
    |------------------------------------|--------   |------------------------------------     |
    | Timestamp Present                  | Pass/Fail | Explanation of missing values, if any  |
    | Correct Timestamp Order            | Pass/Fail | Explanation of timestamp order check   |
    | No Weekend Entries                 | Pass/Fail | Explanation of weekend check           |
    | No Public Holidays                 | Pass/Fail | Explanation of public holiday check    |
    | 4-Digit Ticker Symbol              | Pass/Fail | Explanation of ticker validity         |
    | Valid Ticker Names                 | Pass/Fail | Explanation of recognized tickers      |

    """

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    
    return response['choices'][0]['message']['content'] if response else "No response generated."

# Initialize Streamlit session state
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = []

st.set_page_config(page_title="Search & Validate Stocks", layout="wide")

st.sidebar.title("🔍 Search & Validate Stocks")
ticker = st.sidebar.text_input("📊 Enter Stock Ticker or Company Name")
num_results = st.sidebar.slider("📈 Number of Results", 1, 50, 5)
user_query = st.sidebar.text_input("💡 Ask AI about stocks")
use_search_context = st.sidebar.checkbox("📌 Use Search Context", value=True)

st.markdown("<h1 style='text-align: center; font-weight: 600;'>📈 Stock Search & Validation</h1>", unsafe_allow_html=True)

if st.sidebar.button("🚀 Search"):
    with st.spinner("Fetching results ..."):
        results, error = get_bing_results(ticker, num_results)
        
        if error:
            st.warning(error)
        
        scraped_results = []
        for result in results:
            try:
                page_data = scrape_full_page(result["link"])
                page_data.update({"title": result["title"], "link": result["link"], "snippet": result["snippet"]})
                scraped_results.append(page_data)
            except Exception as e:
                st.warning(f"Error scraping {result['link']}: {e}")
        
        st.session_state.scraped_data = scraped_results
    
    st.success("✅ Search Completed!")

# Display search results
if st.session_state.scraped_data:
    st.subheader("🔍 Scraped Search Results:")
    for item in st.session_state.scraped_data:
        st.markdown(f"### [{item['title']}]({item['link']})")
        st.write(f"📅 Date Scraped: {item['date']}")
        st.write(f"🔎 Snippet: {item['snippet']}")
        st.write(f"🏷️ Ticker: {item['ticker']} | 💲 Price: {item['price']}")
        if 'error' in item:
            st.warning(f"⚠️ Error scraping: {item['error']}")
        st.markdown("---")

# AI Response Section
if user_query and st.session_state.scraped_data:
    with st.spinner("Generating AI response..."):
        response = query_openai_llm(user_query, st.session_state.scraped_data)
        st.subheader("💡 AI Response:")
        st.write(response)

# Validation Section
if st.sidebar.button("✅ Validate Results"):
    if st.session_state.scraped_data:
        with st.spinner("Validating results..."):
            validation_result = validate_results(st.session_state.scraped_data)
            st.subheader("🔍 Validation Results:")
            st.markdown(validation_result)
    else:
        st.warning("⚠️ No data to validate. Please perform a search first.")
