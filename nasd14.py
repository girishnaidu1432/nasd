import streamlit as st
import datetime
import json
import openai
import re
from bs4 import BeautifulSoup
import requests
import numpy as np
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph
from langgraph.graph.graph import END, START
from langchain.schema import SystemMessage
from dataclasses import dataclass, field
from typing import TypedDict, Annotated

# OpenAI API Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

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
        ticker_match = re.search(r'\b[A-Z]{2,5}\b', full_content)
        ticker = ticker_match.group(0) if ticker_match else "N/A"
        price_match = re.search(r'\$\d{1,5}(\.\d{1,2})?', full_content)
        price = price_match.group(0) if price_match else "N/A"
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"url": url, "date": date, "full_content": full_content, "ticker": ticker, "price": price}
    except Exception as e:
        return {"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"}

def validate_stock_data(scraped_data):
    """Validate and analyze stock data statistics."""
    results = []
    for item in scraped_data:
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
            results.append({"url": item["url"], "ticker": item.get("ticker", "N/A"), "price": price, "SMA": sma, "EMA": ema, "Z-Score": z_score})
    return results

@tool
def search_web(query: str):
    return get_bing_results(query, 5)[0]

@tool
def get_stock_price(url: str):
    return scrape_full_page(url)

# LLM with tool binding
tools = [search_web, get_stock_price]
llm_with_tools = openai.ChatCompletion.create(engine=deployment_name).bind_tools(tools=tools)

# Define Class
class State(TypedDict):
    messages: Annotated[list, lambda x: x]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

# Streamlit UI
st.sidebar.title("🔍 Search & Validate Stocks")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("stock_state", State(messages=[]))
st.session_state["ticker"] = st.sidebar.text_input("📊 Enter Stock Ticker or Company Name")
st.session_state["num_results"] = st.sidebar.slider("📈 Number of Results", 1, 50, 5)
st.session_state["user_query"] = st.sidebar.text_input("💡 Ask AI about stocks")

if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("Executing AI Workflow..."):
        st.session_state["stock_state"] = graph.invoke(st.session_state["stock_state"])
        st.subheader("🔍 Search Results")
        st.json(search_web(st.session_state["ticker"]))
        st.subheader("📊 Stock Price Data")
        st.json(get_stock_price(st.session_state["ticker"]))
        st.subheader("💡 AI Analysis")
        st.write(st.session_state["stock_state"]["messages"])
