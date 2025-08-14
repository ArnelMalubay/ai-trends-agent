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
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
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


class SearchQueryParser(BaseOutputParser):
    """Parser for extracting search queries from LLM output."""
    
    def parse(self, text: str) -> List[str]:
        """Parse the LLM output to extract search queries.
        
        Args:
            text: Raw LLM output text
            
        Returns:
            List of search query strings
        """
        try:
            # Try to parse as JSON first
            if text.strip().startswith('[') and text.strip().endswith(']'):
                return json.loads(text)
            
            # Fallback: extract queries from numbered list or bullet points
            queries = []
            lines = text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove numbering and bullet points
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                clean_line = re.sub(r'^[-*]\s*', '', clean_line)
                clean_line = clean_line.strip('"\'')
                
                if clean_line and len(clean_line) > 3:
                    queries.append(clean_line)
                    
            return queries[:10]  # Limit to 10 queries max
            
        except Exception as e:
            logger.error(f"Error parsing search queries: {e}")
            return [text.strip()]


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


def generate_search_queries(text_input: str, k: int = 5) -> List[str]:
    """Generate k best search queries for finding relevant AI/data science content.
    
    This function uses an LLM to analyze the input text and generate optimized
    search queries that will help find the most relevant articles, websites,
    and news related to AI trends and data science.
    
    Args:
        text_input: The input text describing the search intent
        k: Number of search queries to generate (default: 5)
        
    Returns:
        List of k search query strings optimized for finding relevant content
        
    Raises:
        Exception: If LLM initialization or query generation fails
    """
    try:
        llm = initialize_llm()
        
        prompt_template = PromptTemplate(
            input_variables = ["text_input", "k"],
            template = """
You are an expert at creating search queries for finding the latest AI and data science trends, news, and announcements.

Given the following input text, generate {k} highly effective search queries that will help find the most relevant and recent articles, websites, blog posts, and news related to AI trends and data science.

Input text: "{text_input}"

Guidelines for creating search queries:
1. Focus on recent developments and trends
2. Include specific AI/ML/data science terminology
3. Consider different aspects (research, industry news, tools, breakthroughs)
4. Use keywords that news sites and tech blogs commonly use
5. Include time-sensitive terms like "2024", "latest", "new", "breakthrough"

Your response should EXACTLY follow the format:
[
    "query 1",
    "query 2",
    "query 3"
]

without any introduction or explanation.
"""
        )
        
        chain = prompt_template | llm | SearchQueryParser()
        queries = chain.invoke({
            "text_input": text_input,
            "k": k
        })
        
        # Ensure we have exactly k queries
        if len(queries) < k:
            # Pad with variations of the input
            base_query = text_input.strip()
            while len(queries) < k:
                queries.append(f"{base_query} latest news")
                queries.append(f"{base_query} trends 2024")
                queries.append(f"{base_query} breakthrough")
                
        return queries[:k]
        
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # Fallback to simple queries
        return [
            f"{text_input} AI trends",
            f"{text_input} latest news",
            f"{text_input} data science",
            f"{text_input} machine learning",
            f"{text_input} artificial intelligence"
        ][:k]


def extract_text_from_url(url: str, max_length: int = 500) -> str:
    """Extract and summarize text content from a URL.
    
    Args:
        url: The URL to extract text from
        max_length: Maximum length of extracted text
        
    Returns:
        Extracted and cleaned text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers = headers, timeout = 10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extract text from paragraphs and headings
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = ' '.join([elem.get_text().strip() for elem in text_elements])
        
        # Clean and truncate text
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_length] + "..." if len(text) > max_length else text
        
    except Exception as e:
        logger.warning(f"Could not extract text from {url}: {e}")
        return ""


def search_web_content(search_query: str, k: int = 5) -> List[LinkSummary]:
    """Search the web using Google Search API and return k [link, summary] pairs.
    
    This function searches the internet using Google Search API and returns
    relevant articles, websites, and news with their summaries.
    
    Args:
        search_query: The search query to use
        k: Number of results to return (default: 5)
        
    Returns:
        List of LinkSummary objects containing links and summaries
        
    Note:
        Requires GOOGLE_CSE_ID and GOOGLE_API_KEY environment variables
    """
    try:        
        # Initialize Google search
        search = GoogleSearchAPIWrapper(k = k * 2)  # Get more results to filter
        
        # Perform search
        results = search.results(search_query, k * 2)
        
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

Return ONLY a JSON array of {num_items} numerical scores (0.0 to 1.0), like:
[0.9, 0.7, 0.3, 0.8, 0.5]

Relevance scores:
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


def get_relevant_ai_trends(text_input: str, k: int = 5) -> List[LinkSummary]:
    """Get the top k most relevant [link, summary] pairs for AI trends.
    
    This is the main orchestration function that combines all other functions
    to provide end-to-end search and ranking for AI and data science content.
    
    Args:
        text_input: The input text describing what to search for
        k: Number of top results to return (default: 5)
        
    Returns:
        List of top k most relevant LinkSummary objects for AI trends
    """
    try:
        logger.info(f"Starting AI trends search for: {text_input}")
        
        # Step 1: Generate optimized search queries
        logger.info("Generating search queries...")
        search_queries = generate_search_queries(text_input, k = 3)
        logger.info(f"Generated {len(search_queries)} search queries")
        
        # Step 2: Search web content for each query
        logger.info("Searching web content...")
        all_link_summaries = []
        
        for query in search_queries:
            results = search_web_content(query, k = k)
            all_link_summaries.extend(results)
            logger.info(f"Found {len(results)} results for query: {query}")
            
        # Remove duplicates based on URL
        unique_summaries = []
        seen_links = set()
        
        for ls in all_link_summaries:
            if ls.link not in seen_links:
                unique_summaries.append(ls)
                seen_links.add(ls.link)
                
        logger.info(f"Found {len(unique_summaries)} unique results")
        
        # Step 3: Rank by relevance
        logger.info("Ranking results by relevance...")
        ranked_results = rank_by_relevance(text_input, unique_summaries, k = k)
        
        logger.info(f"Returning top {len(ranked_results)} most relevant results")
        return ranked_results
        
    except Exception as e:
        logger.error(f"Error in get_relevant_ai_trends: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # # Test the main function
    # test_input = "Philippines AI trends"
    
    # print(f"Searching for: {test_input}")
    # print("=" * 50)
    
    # results = get_relevant_ai_trends(test_input, k = 3)
    
    # for i, result in enumerate(results, 1):
    #     print(f"\n{i}. Relevance Score: {result.relevance_score:.2f}")
    #     print(f"   Link: {result.link}")
    #     print(f"   Summary: {result.summary[:200]}...")
    #     print("-" * 50)

    test_input = "Philippines AI trends this year"
    results = generate_search_queries(test_input, k = 2)
    print('-' * 30)
    for result in results:
        print(result)
        print('-' * 30)
    web_results = search_web_content(results[1], k = 2)
    print('-' * 30)
    for result in web_results:
        print(result.link)
        print(result.summary)
        print('-' * 30)