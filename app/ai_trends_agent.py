"""
AI Trends Agent - LangChain-based agent for discovering AI and data science trends.

This module provides functions to search for and rank relevant articles, websites,
and news related to AI trends using LLM-powered search query generation and
web search capabilities.
"""

import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import requests
import re
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_google_community import GoogleSearchAPIWrapper
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LinkSummary:
    """Data class to represent a link and its summary."""
    link: str
    summary: str
    relevance_score: Optional[float] = None


class RelevanceScoreParser(BaseOutputParser):
    """Parser for extracting relevance scores from LLM output."""
    
    def parse(self, text: str) -> List[float]:
        """Parse the LLM output to extract relevance scores.
        
        Args:
            text: Raw LLM output text containing scores
            
        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        try:
            # Try to parse as JSON array first
            if text.strip().startswith('[') and text.strip().endswith(']'):
                scores = json.loads(text)
                return [float(score) for score in scores]
            
            # Extract numbers from text
            numbers = re.findall(r'\b(?:0\.\d+|1\.0+|0|1)\b', text)
            scores = [float(num) for num in numbers]
            
            # Normalize scores to 0-1 range if needed
            normalized_scores = []
            for score in scores:
                if score > 1.0:
                    score = score / 10.0  # Assume 0-10 scale
                normalized_scores.append(max(0.0, min(1.0, score)))
                
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error parsing relevance scores: {e}")
            return [0.5] * 10  # Return default scores


def initialize_llm() -> ChatGroq:
    """Initialize the Groq LLM client.
    
    Returns:
        Configured ChatGroq instance
        
    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    return ChatGroq(
        groq_api_key = api_key,
        model_name = "llama3-70b-8192",  # High-quality model for search queries
        temperature = 0.3,  # Lower temperature for more focused outputs
        max_tokens = 2000
    )



def extract_text_from_url(url: str, max_length: int = 500) -> str:
    """Extract and summarize text content from a URL using LangChain's UnstructuredURLLoader.
    
    Args:
        url: The URL to extract text from
        max_length: Maximum length of extracted text
        
    Returns:
        Extracted and cleaned text content, or empty string if error content detected
    """
    try:
        # Use LangChain's UnstructuredURLLoader
        loader = UnstructuredURLLoader(urls = [url])
        docs = loader.load()
        
        if docs and len(docs) > 0:
            # Get the text content from the first document
            text = docs[0].page_content
            
            # Clean and normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Check for error content patterns (case-insensitive)
            error_patterns = [
                r'403\s*forbidden',
                r'404\s*not\s*found',
                r'500\s*internal\s*server\s*error',
                r'502\s*bad\s*gateway',
                r'503\s*service\s*unavailable',
                r'504\s*gateway\s*timeout',
                r'nginx.*error',
                r'apache.*error',
                r'server\s*error',
                r'access\s*denied',
                r'permission\s*denied',
                r'cloudflare.*error',
                r'this\s*page\s*is\s*not\s*available',
                r'page\s*not\s*found',
                r'website\s*is\s*temporarily\s*unavailable',
                r'maintenance\s*mode',
                r'blocked\s*by\s*administrator'
            ]
            
            # Check if text contains error patterns
            text_lower = text.lower()
            for pattern in error_patterns:
                if re.search(pattern, text_lower):
                    logger.warning(f"Error content detected from {url}: {pattern}")
                    return ""
            
            # Additional check: if text is very short and contains common error keywords
            if len(text) < 100 and any(keyword in text_lower for keyword in 
                ['error', 'forbidden', 'denied', 'unavailable', 'blocked']):
                logger.warning(f"Short error content detected from {url}")
                return ""
            
            # Truncate if necessary
            return text[:max_length] + "..." if len(text) > max_length else text
        else:
            logger.warning(f"No content extracted from {url}")
            return ""
        
    except Exception as e:
        logger.warning(f"Could not extract text from {url}: {e}")
        return ""


def search_web_content(search_query: str, k: int = 5, region: str = "ph", date_restrict: str = 'd7') -> List[LinkSummary]:
    """Search the web using Google Search API and return k [link, summary] pairs.
    
    This function searches the internet using Google Search API and returns
    relevant articles, websites, and news with their summaries.
    
    Args:
        search_query: The search query to use
        k: Number of results to return (default: 5)
        region: Country code for regional search results (default: "ph" for Philippines)
        date_restrict: Date restriction for results (e.g., "w1" = last week, "d7" = last 7 days, "m1" = last month)
        
    Returns:
        List of LinkSummary objects containing links and summaries
        
    Note:
        Requires GOOGLE_CSE_ID and GOOGLE_API_KEY environment variables
        Common region codes: "ph" (Philippines), "us" (USA), "uk" (UK), "sg" (Singapore)
        Date restrict options: "d1", "d7", "w1", "m1", "m3", "m6", "y1"
    """
    try:        
        # Initialize Google search
        search = GoogleSearchAPIWrapper(k = k * 2)  # Get more results to filter
        
        # Enhanced search parameters to match web interface behavior
        search_params = {
            "gl": region,       # Geolocation
            "lr": "lang_en",    # Language restriction
            "safe": "medium",   # Safe search (off, medium, high)
            "filter": "1",      # Include similar results (0=exclude, 1=include)
        }
        
        # Add date restriction if specified
        if date_restrict:
            search_params["dateRestrict"] = date_restrict
        
        # Debug logging to see what's being sent to the API
        logger.info(f"Search query: '{search_query}'")
        logger.info(f"Search params: {search_params}")
        logger.info(f"Expected results: {k * 2}")
        
        results = search.results(search_query, k * 2, search_params = search_params)
        logger.info(f"Actual results returned: {len(results)}")
        
        link_summaries = []
        
        for result in results[:k]:
            # Extract basic info from search result
            link = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            if not link:
                continue
                
            # Try to get more detailed content
            detailed_content = extract_text_from_url(link)
            
            # Create summary combining title, snippet, and extracted content
            if detailed_content:
                # Full summary with extracted content
                summary_parts = []
                if title:
                    summary_parts.append(f"Title: {title}")
                if snippet:
                    summary_parts.append(f"Snippet: {snippet}")
                summary_parts.append(f"Content: {detailed_content}")
                summary = " | ".join(summary_parts)
            else:
                # Fallback format when content extraction fails
                summary = f"Title: {title} :: Snippet: {snippet}"
            
            link_summaries.append(LinkSummary(
                link = link,
                summary = summary
            ))
            
        return link_summaries[:k]
        
    except Exception as e:
        logger.error(f"Error searching web content: {e}")
        return []


def rank_by_relevance(
    text_input: str, 
    link_summaries: List[LinkSummary], 
    k: int = 5
) -> List[LinkSummary]:
    """Rank [link, summary] pairs by relevance to the input text.
    
    This function uses an LLM to assess the relevance of each link and summary
    to the original text input and returns the top k most relevant pairs.
    
    Args:
        text_input: The original text input to compare against
        link_summaries: List of LinkSummary objects to rank
        k: Number of top results to return (default: 5)
        
    Returns:
        List of top k most relevant LinkSummary objects, sorted by relevance
    """
    if not link_summaries:
        return []
        
    try:
        llm = initialize_llm()
        
        # Prepare content for LLM evaluation
        content_for_ranking = []
        for i, ls in enumerate(link_summaries):
            content_for_ranking.append(f"{i+1}. {ls.summary[:300]}...")
            
        content_text = "\n\n".join(content_for_ranking)
        
        prompt_template = PromptTemplate(
            input_variables = ["text_input", "content_text", "num_items"],
            template = """
You are an expert at assessing the relevance of content to specific topics in AI and data science.

Original search intent: "{text_input}"

Please rate the relevance of each of the following {num_items} pieces of content to the original search intent. Rate each on a scale from 0.0 to 1.0, where:
- 1.0 = Extremely relevant and directly addresses the search intent
- 0.8 = Highly relevant with strong connections
- 0.6 = Moderately relevant 
- 0.4 = Somewhat relevant
- 0.2 = Minimally relevant
- 0.0 = Not relevant at all

Content to evaluate:
{content_text}

Return EXACTLY an array of {num_items} numerical scores (0.0 to 1.0), following the format below:
[0.9, 0.7, 0.3, 0.8, 0.5]
"""
        )
        
        chain = prompt_template | llm | RelevanceScoreParser()
        scores = chain.invoke({
            "text_input": text_input,
            "content_text": content_text,
            "num_items": len(link_summaries)
        })
        
        # Assign scores to link summaries
        for i, score in enumerate(scores):
            if i < len(link_summaries):
                link_summaries[i].relevance_score = score
                
        # Sort by relevance score (descending) and return top k
        ranked_summaries = sorted(
            link_summaries, 
            key = lambda x: x.relevance_score or 0.0, 
            reverse = True
        )
        
        return ranked_summaries[:k]
        
    except Exception as e:
        logger.error(f"Error ranking content by relevance: {e}")
        # Return original list truncated to k items
        return link_summaries[:k]


def get_relevant_ai_trends(search_queries: List[str], ranking_context: str = "", k: int = 5, region: str = "ph", date_restrict: str = 'd7') -> dict[str, List[LinkSummary]]:
    """Get the top k most relevant [link, summary] pairs for each search query.
    
    This function processes each search query individually, searching and ranking 
    results separately for each query.
    
    Args:
        search_queries: List of search query strings to use
        ranking_context: Optional context text for relevance ranking (if empty, uses each query)
        k: Number of top results to return per query (default: 5)
        region: Country code for regional search results (default: "ph" for Philippines)
        date_restrict: Date restriction for results (e.g., "w1" = last week, "d7" = last 7 days)
        
    Returns:
        Dictionary where keys are search queries and values are lists of top k LinkSummary objects
    """
    try:
        logger.info(f"Starting AI trends search with {len(search_queries)} queries")
        
        results_by_query = {}
        
        # Process each search query individually
        for query in search_queries:
            logger.info(f"Processing query: '{query}'")
            
            try:
                # Step 1: Search web content for this specific query
                logger.info(f"Searching web content for: {query}")
                query_results = search_web_content(query, k = k * 2, region = region, date_restrict = date_restrict)  # Get more results to rank
                logger.info(f"Found {len(query_results)} results for query: {query}")
                
                if query_results:
                    # Step 2: Rank results by relevance for this specific query
                    logger.info(f"Ranking results for query: {query}")
                    # Use ranking_context if provided, otherwise use the current query
                    context_for_ranking = ranking_context if ranking_context else query
                    ranked_results = rank_by_relevance(context_for_ranking, query_results, k = k)
                    
                    results_by_query[query] = ranked_results
                    logger.info(f"Returning top {len(ranked_results)} results for query: {query}")
                else:
                    results_by_query[query] = []
                    logger.warning(f"No results found for query: {query}")
                    
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results_by_query[query] = []
        
        logger.info(f"Completed processing all {len(search_queries)} queries")
        return results_by_query
        
    except Exception as e:
        logger.error(f"Error in get_relevant_ai_trends: {e}")
        return {}


# Example usage and testing
if __name__ == "__main__":
    # Test with static search queries (new approach)
    search_query = "latest ai news in the philippines"
    results = search_web_content(search_query, k = 10, region = "ph", date_restrict = "w1")
    for result in results:
        print(result)
        print('-' * 30)
    # search_queries = [
    #     "latest ai news in the philippines",
    #     "artificial intelligence trends philippines 2024",
    #     "machine learning developments manila"
    # ]
    
    # print("Testing AI Trends Agent with Static Queries")
    # print("=" * 50)
    # print(f"Search queries: {search_queries}")
    
    # # Test the main orchestration function with last week filter
    # results_dict = get_relevant_ai_trends(
    #     search_queries = search_queries,
    #     ranking_context = "AI trends and developments in the Philippines",
    #     k = 3,
    #     region = "ph",
    #     date_restrict = "w1"  # Last week only
    # )
    
    # print(f"\nResults by Query:")
    # print("=" * 50)
    
    # for query, results in results_dict.items():
    #     print(f"\n🔍 Query: '{query}'")
    #     print(f"📊 Found {len(results)} results:")
    #     print("-" * 30)
        
    #     for i, result in enumerate(results, 1):
    #         print(f"{i}. Relevance: {result.relevance_score:.2f}")
    #         print(f"   Link: {result.link}")
    #         print(f"   Summary: {result.summary[:150]}...")
    #         print()
        
    #     print("=" * 50)