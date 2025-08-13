"""
Weekly AI Trends Scheduler

This script can be run weekly to automatically gather AI trends and send an email report.
Set up a cron job or scheduled task to run this script automatically.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List
from ai_trends_agent import get_relevant_ai_trends, LinkSummary
import logging

# Configure logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def format_html_report(results: List[LinkSummary], search_topic: str) -> str:
    """Format the search results into an HTML email report.
    
    Args:
        results: List of LinkSummary objects
        search_topic: The search topic used
        
    Returns:
        HTML formatted email content
    """
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            .result {{ 
                border: 1px solid #bdc3c7; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f8f9fa;
            }}
            .relevance {{ 
                background-color: #e74c3c; 
                color: white; 
                padding: 3px 8px; 
                border-radius: 3px; 
                font-size: 0.8em;
            }}
            .link {{ color: #3498db; text-decoration: none; }}
            .link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>🤖 Weekly AI Trends Report</h1>
        <p><strong>Generated on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <p><strong>Search Topic:</strong> {search_topic}</p>
        
        <h2>Top {len(results)} Most Relevant Findings:</h2>
    """
    
    for i, result in enumerate(results, 1):
        relevance_score = result.relevance_score or 0.0
        relevance_color = "#e74c3c" if relevance_score >= 0.8 else "#f39c12" if relevance_score >= 0.6 else "#95a5a6"
        
        html_content += f"""
        <div class="result">
            <h3>{i}. <a href="{result.link}" class="link" target="_blank">Link to Article</a>
            <span class="relevance" style="background-color: {relevance_color}">
                Relevance: {relevance_score:.1f}/1.0
            </span></h3>
            <p><strong>Summary:</strong> {result.summary[:400]}{'...' if len(result.summary) > 400 else ''}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content


def send_email_report(
    html_content: str, 
    subject: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    username: str = None,
    password: str = None,
    recipient: str = None
) -> bool:
    """Send the HTML report via email.
    
    Args:
        html_content: HTML formatted email content
        subject: Email subject line
        smtp_server: SMTP server address
        smtp_port: SMTP server port
        username: Email username
        password: Email password
        recipient: Recipient email address
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = username
        message["To"] = recipient
        
        # Attach HTML content
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, recipient, message.as_string())
            
        logger.info(f"Email report sent successfully to {recipient}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def run_weekly_report(
    search_topics: List[str] = None,
    num_results: int = 5,
    send_email: bool = True
) -> None:
    """Run the weekly AI trends report.
    
    Args:
        search_topics: List of topics to search for
        num_results: Number of results per topic
        send_email: Whether to send email report
    """
    if search_topics is None:
        search_topics = [
            "latest AI breakthroughs and research 2024",
            "new machine learning tools and frameworks",
            "AI industry news and product launches",
            "data science trends and developments"
        ]
    
    logger.info("Starting weekly AI trends report generation...")
    
    all_results = []
    
    for topic in search_topics:
        logger.info(f"Searching for: {topic}")
        try:
            results = get_relevant_ai_trends(topic, k = num_results)
            all_results.extend(results)
            logger.info(f"Found {len(results)} results for topic: {topic}")
        except Exception as e:
            logger.error(f"Error searching for topic '{topic}': {e}")
    
    if not all_results:
        logger.warning("No results found for any topic")
        return
    
    # Remove duplicates and sort by relevance
    unique_results = {}
    for result in all_results:
        if result.link not in unique_results:
            unique_results[result.link] = result
        elif result.relevance_score and result.relevance_score > unique_results[result.link].relevance_score:
            unique_results[result.link] = result
    
    final_results = sorted(
        unique_results.values(),
        key = lambda x: x.relevance_score or 0.0,
        reverse = True
    )[:15]  # Top 15 results
    
    logger.info(f"Generated report with {len(final_results)} unique results")
    
    # Generate HTML report
    search_summary = " | ".join(search_topics)
    html_report = format_html_report(final_results, search_summary)
    
    # Save report locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ai_trends_report_{timestamp}.html"
    
    with open(report_filename, 'w', encoding = 'utf-8') as f:
        f.write(html_report)
    
    logger.info(f"Report saved as {report_filename}")
    
    # Send email if configured
    if send_email:
        username = os.getenv('EMAIL_USERNAME')
        password = os.getenv('EMAIL_PASSWORD')
        recipient = os.getenv('EMAIL_RECIPIENT')
        smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        
        if username and password and recipient:
            subject = f"🤖 Weekly AI Trends Report - {datetime.now().strftime('%B %d, %Y')}"
            
            success = send_email_report(
                html_report,
                subject,
                smtp_server,
                smtp_port,
                username,
                password,
                recipient
            )
            
            if success:
                logger.info("Weekly report sent successfully!")
            else:
                logger.error("Failed to send weekly report")
        else:
            logger.warning("Email configuration not found. Report saved locally only.")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the weekly report
    run_weekly_report()
