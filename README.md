# AI Trends Agent

A LangChain-powered agent that automatically discovers and ranks the latest trends, news, and announcements in AI and data science. Perfect for staying up-to-date with the rapidly evolving AI landscape.

## Features

- 🤖 **LLM-Powered Search**: Uses Groq's LLaMA models to generate optimized search queries
- 🔍 **Web Search Integration**: Searches the web for relevant articles and news
- 📊 **Intelligent Ranking**: Ranks results by relevance using AI
- 📧 **Weekly Reports**: Automated email reports with the latest AI trends
- 🎯 **Customizable**: Easy to customize search topics and output formats

## Quick Start

1. **Clone and Install**
   ```bash
   git clone <your-repo-url>
   cd ai-trends-agent
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Create .env file with your Groq API key
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

3. **Test the Agent**
   ```bash
   cd app
   python ai_trends_agent.py
   ```

4. **Run Weekly Report**
   ```bash
   python weekly_scheduler.py
   ```

## Core Functions

The agent provides four main functions:

1. **`generate_search_queries(text_input, k=5)`** - Generates k optimized search queries from text input
2. **`search_web_content(search_query, k=5)`** - Returns k [link, summary] pairs from web search
3. **`rank_by_relevance(text_input, link_summaries, k=5)`** - Ranks results by relevance to input
4. **`get_relevant_ai_trends(text_input, k=5)`** - Complete pipeline combining all functions

## Configuration

See [app/CONFIG.md](app/CONFIG.md) for detailed setup instructions including:
- Groq API key setup
- Email configuration for weekly reports  
- Scheduling automation
- Customization options

## Example Usage

```python
from app.ai_trends_agent import get_relevant_ai_trends

# Get top 5 results for AI trends
results = get_relevant_ai_trends("latest developments in large language models", k=5)

for result in results:
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"Link: {result.link}")
    print(f"Summary: {result.summary[:150]}...")
    print("-" * 50)
```

## Project Structure

```
ai-trends-agent/
├── app/
│   ├── ai_trends_agent.py      # Core agent functions
│   ├── weekly_scheduler.py     # Weekly report automation
│   └── CONFIG.md              # Detailed configuration guide
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
└── README.md                 # This file
```

## Technologies Used

- **LangChain**: LLM orchestration and prompt management
- **Groq**: Fast LLM inference with LLaMA models
- **DuckDuckGo**: Web search API
- **BeautifulSoup**: Web content extraction
- **SMTP**: Email report delivery

## License

MIT License - Feel free to use and modify for your needs!
