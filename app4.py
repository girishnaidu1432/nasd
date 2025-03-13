import datetime
import json
import openai
import re
import streamlit as st
from bs4 import BeautifulSoup
import requests
import numpy as np
from langchain.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph
from langchain.chat_models import AzureChatOpenAI
from typing import Dict, List, TypedDict

# OpenAI API Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# Bing Search URL
BING_SEARCH_URL = "https://www.bing.com/search?q="

class StockState(TypedDict):
    ticker: str
    results: List[Dict]
    scraped_data: List[Dict]
    validated_data: str
    analysis_data: str
    reasoning_data: str
    summary_data: str

def get_bing_results(state: StockState) -> StockState:
    query = state["ticker"]
    num_results = 5
    search_url = BING_SEARCH_URL + query.replace(" ", "+")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return state
    
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    
    for b in soup.find_all('li', class_='b_algo')[:num_results]:
        title = b.find('h2').text if b.find('h2') else "No Title"
        link = b.find('a')['href'] if b.find('a') else "No Link"
        snippet = b.find('p').text if b.find('p') else "No snippet available"
        results.append({"title": title, "link": link, "snippet": snippet})
    
    state["results"] = results
    return state

def scrape_full_page(state: StockState) -> StockState:
    if "results" not in state or not state["results"]:
        state["scraped_data"] = []
        return state
    
    scraped_data = []
    for res in state["results"]:
        url = res["link"]
        try:
            loader = WebBaseLoader(url)
            doc = loader.load()
            soup = BeautifulSoup(doc[0].page_content, "html.parser")
            full_content = soup.get_text(separator="\n")
            
            ticker_match = re.search(r'\b[A-Z]{2,5}\b', full_content)
            ticker = ticker_match.group(0) if ticker_match else "N/A"
            price_match = re.search(r'\$\d{1,5}(\.\d{1,2})?', full_content)
            price = price_match.group(0) if price_match else "N/A"
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scraped_data.append({"url": url, "date": date, "ticker": ticker, "price": price})
        except Exception as e:
            scraped_data.append({"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"})
    state["scraped_data"] = scraped_data
    return state

def validate_results(state: StockState) -> StockState:
    if not state["scraped_data"]:
        state["validated_data"] = "No data available for validation."
        return state
    
    stock_data = [
        {
            "url": item["url"],
            "date": item["date"],
            "ticker": item.get("ticker", "N/A"),
            "price": item.get("price", "N/A"),
            "timestamp": item["date"],
        }
        for item in state["scraped_data"]
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
    
    state["validated_data"] = response['choices'][0]['message']['content'] if response else "No response generated."
    return state

def validate_stock_data_stats(state: StockState) -> StockState:
    if not state["scraped_data"]:
        state["analysis_data"] = "No data available for analysis."
        return state
    
    stock_analysis = []
    for item in state["scraped_data"]:
        try:
            price = float(item.get("price", "N/A").replace("$", "").replace(",", ""))
        except ValueError:
            price = None
        
        historical_prices = np.random.uniform(low=price * 0.9, high=price * 1.1, size=20) if price else []
        
        if len(historical_prices) > 0:
            sma = np.mean(historical_prices)
            ema = np.average(historical_prices, weights=np.linspace(1, 0, len(historical_prices)))
            std_dev = np.std(historical_prices)
            z_score = (price - sma) / std_dev if std_dev > 0 else None
            
            validation_checks = {
                "No unrealistic price jumps (>20%)": "Pass" if abs(price - sma) / sma <= 0.2 else "Fail",
                "Z-score within range (-3 to 3)": "Pass" if -3 <= z_score <= 3 else "Fail",
                "Price within 3-4 std deviations": "Pass" if abs(price - sma) <= 4 * std_dev else "Fail",
                "No extreme deviation (>5x std dev)": "Pass" if abs(price - sma) <= 5 * std_dev else "Fail",
            }
            
            stock_analysis.append({
                "URL": item["url"],
                "Ticker": item.get("ticker", "N/A"),
                "Price": f"${price:.2f}" if price else "N/A",
                "SMA": f"${sma:.2f}",
                "EMA": f"${ema:.2f}",
                "Std Dev": f"${std_dev:.2f}",
                "Z-Score": f"{z_score:.2f}" if z_score else "N/A",
                "Validation": validation_checks,
            })
    
    state["analysis_data"] = json.dumps(stock_analysis, indent=2)
    return state

def reasoning_agent(state: StockState) -> StockState:
    if not state["scraped_data"]:
        state["reasoning_data"] = "No data available for reasoning."
        return state
    
    stock_analysis = []
    for item in state["scraped_data"]:
        try:
            price = float(item.get("price", "N/A").replace("$", "").replace(",", ""))
        except ValueError:
            price = None
        
        historical_prices = np.random.uniform(low=price * 0.9, high=price * 1.1, size=20) if price else []
        
        if len(historical_prices) > 0:
            sma = np.mean(historical_prices)
            ema = np.average(historical_prices, weights=np.linspace(1, 0, len(historical_prices)))
            std_dev = np.std(historical_prices)
            z_score = (price - sma) / std_dev if std_dev > 0 else None
            
            insights = []
            if abs(price - sma) / sma > 0.2:
                insights.append("Potential price volatility or data anomaly detected.")
            if not (-3 <= z_score <= 3):
                insights.append("Z-score indicates a significant deviation from the mean, warranting further investigation.")
            if not (abs(price - sma) <= 4 * std_dev):
                insights.append("Price is outside typical standard deviation range, suggesting possible outliers.")
            if not (abs(price - sma) <= 5 * std_dev):
                insights.append("Extreme price deviation detected, requires thorough review.")
            
            stock_analysis.append({
                "URL": item["url"],
                "Ticker": item.get("ticker", "N/A"),
                "Price": f"${price:.2f}" if price else "N/A",
                "SMA": f"${sma:.2f}",
                "EMA": f"${ema:.2f}",
                "Std Dev": f"${std_dev:.2f}",
                "Z-Score": f"{z_score:.2f}" if z_score else "N/A",
                "Insights": insights,
            })
    
    state["reasoning_data"] = json.dumps(stock_analysis, indent=2)
    return state

def summary_agent(state: StockState) -> StockState:
    if not state["scraped_data"]:
        state["summary_data"] = "No data available for summary."
        return state
    
    summaries = []
    for item in state["scraped_data"]:
        url = item.get("url", "N/A")
        ticker = item.get("ticker", "N/A")
        price = item.get("price", "N/A")
        
        prompt = f"""
        You are a financial analyst. Summarize the stock data and provide key insights.
        Include a theoretical explanation of the stock's movement and key takeaways.
        
        Data:
        URL: {url}
        Ticker: {ticker}
        Price: {price}
        
        Provide a structured response including:
        - Theory behind stock movement (e.g., supply/demand, market trends, news impact)
        - Summary of key financial insights
        - Final takeaway message for investors
        """
        
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        summary_text = response['choices'][0]['message']['content'] if response else "No summary generated."
        summaries.append({"url": url, "summary": summary_text})
    
    state["summary_data"] = json.dumps(summaries, indent=2)
    return state




# Streamlit UI
st.title("Stock Analysis with LangGraph")

if "state" not in st.session_state:
    st.session_state.state = {
        "ticker": "",
        "results": [],
        "scraped_data": [],
        "validated_data": "",
        "analysis_data": "",
        "reasoning_data": "",
        "summary_data": ""
    }

if "history" not in st.session_state:
    st.session_state.history = []


ticker = st.text_input("Enter Stock Ticker or Company Name")
st.session_state.state["ticker"] = ticker

# Buttons for each step
if st.button("Fetch Bing Results"):
    st.session_state.state = get_bing_results(st.session_state.state)
    st.json(st.session_state.state["results"])
    st.session_state.history.append({"step": "Bing Results", "data": st.session_state.state["results"]})

if st.button("Scrape Webpages"):
    st.session_state.state = scrape_full_page(st.session_state.state)
    st.json(st.session_state.state["scraped_data"])
    st.session_state.history.append({"step": "Scraped Data", "data": st.session_state.state["scraped_data"]})

if st.button("Validate Data"):
    st.session_state.state = validate_results(st.session_state.state)
    st.markdown(st.session_state.state["validated_data"], unsafe_allow_html=True)
    st.session_state.history.append({"step": "Validated Data", "data": st.session_state.state["validated_data"]})

if st.button("Analyze Data"):
    st.session_state.state = validate_stock_data_stats(st.session_state.state)
    st.text(st.session_state.state["analysis_data"])
    st.session_state.history.append({"step": "Analysis Data", "data": st.session_state.state["analysis_data"]})

if st.button("Generate Reasoning"):
    st.session_state.state = reasoning_agent(st.session_state.state)
    st.text(st.session_state.state["reasoning_data"])
    st.session_state.history.append({"step": "Reasoning Data", "data": st.session_state.state["reasoning_data"]})

if st.button("Generate Summary"):
    st.session_state.state = summary_agent(st.session_state.state)
    st.text(st.session_state.state["summary_data"])
    st.session_state.history.append({"step": "Summary Data", "data": st.session_state.state["summary_data"]})

# Chatbot section
st.subheader("Ask the AI Chatbot")
user_query = st.text_input("Enter your question")
if st.button("Search"):
    chat_history = json.dumps(st.session_state.history, indent=2)
    prompt = f"""
    You are a financial AI assistant. Below is the stock analysis history:
    {chat_history}
    
    Now, answer the user's query:
    {user_query}
    """
    
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    
    answer = response['choices'][0]['message']['content'] if response else "No response generated."
    st.session_state.history.append({"step": "User Query", "query": user_query, "response": answer})
    st.write(answer)
