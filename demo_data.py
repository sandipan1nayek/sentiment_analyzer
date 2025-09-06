"""
Demo Data Generator for Sentiment Analysis
Provides realistic sample data when API keys are not available
"""

import random
from datetime import datetime, timedelta
import json

class DemoDataGenerator:
    def __init__(self):
        self.sample_data = {
            'artificial intelligence': {
                'positive': [
                    {"title": "AI Breakthrough in Medical Diagnosis Saves Lives", "description": "Revolutionary AI system achieves 95% accuracy in early cancer detection, potentially saving thousands of lives annually"},
                    {"title": "Tech Giants Invest $50B in AI Research for Good", "description": "Major technology companies announce massive funding for AI solutions to climate change and healthcare"},
                    {"title": "AI Helps Scientists Discover New Antibiotics", "description": "Machine learning algorithms identify promising compounds that could fight drug-resistant bacteria"},
                    {"title": "AI-Powered Education Platform Improves Learning", "description": "Students using AI tutoring systems show 40% improvement in test scores across multiple subjects"},
                    {"title": "Autonomous Vehicles Reduce Traffic Accidents by 60%", "description": "Latest safety data shows AI-driven cars significantly outperform human drivers in preventing collisions"},
                ],
                'negative': [
                    {"title": "AI Job Displacement Concerns Grow Among Workers", "description": "New study suggests 30% of current jobs could be automated within the next decade, raising unemployment fears"},
                    {"title": "AI Facial Recognition Sparks Privacy Debate", "description": "Civil rights groups challenge widespread surveillance technology deployment in public spaces"},
                    {"title": "Biased AI Algorithms Discriminate in Hiring", "description": "Investigation reveals AI recruitment tools show systematic bias against minority candidates"},
                    {"title": "AI-Generated Deepfakes Threaten Democracy", "description": "Experts warn about increasing sophistication of fake videos created by artificial intelligence"},
                    {"title": "AI Systems Vulnerable to Cyber Attacks", "description": "Security researchers demonstrate how hackers can manipulate AI decision-making processes"},
                ],
                'neutral': [
                    {"title": "Global AI Conference Discusses Future Regulations", "description": "Industry leaders and policymakers gather to establish international standards for AI development"},
                    {"title": "AI Research Funding Reaches Record Levels", "description": "Government and private investment in artificial intelligence research hits $100 billion globally"},
                    {"title": "Universities Launch New AI Ethics Programs", "description": "Top academic institutions introduce curricula focused on responsible AI development"},
                    {"title": "AI Market Expected to Double by 2025", "description": "Market analysts project continued growth in artificial intelligence industry across sectors"},
                    {"title": "Tech Companies Form AI Safety Coalition", "description": "Major firms collaborate on establishing best practices for AI development and deployment"},
                ]
            },
            'tesla': {
                'positive': [
                    {"title": "Tesla Reports Record Quarterly Deliveries", "description": "Electric vehicle pioneer exceeds expectations with 500,000 vehicles delivered, stock surges 12%"},
                    {"title": "Tesla Supercharger Network Expands Globally", "description": "Company announces 10,000 new charging stations worldwide, addressing range anxiety concerns"},
                    {"title": "Tesla's Full Self-Driving Beta Shows Promise", "description": "Latest autonomous driving software demonstrates significant improvements in city navigation"},
                    {"title": "Tesla Gigafactory Brings Jobs to Local Community", "description": "New manufacturing facility creates 5,000 jobs and boosts regional economic development"},
                    {"title": "Tesla Solar Roof Installations Accelerate", "description": "Home solar technology adoption increases 300% as costs decrease and efficiency improves"},
                ],
                'negative': [
                    {"title": "Tesla Faces Quality Control Issues", "description": "Consumer reports cite paint defects and panel gaps in recent Model Y deliveries"},
                    {"title": "Tesla Autopilot Involved in Highway Accident", "description": "Federal investigators examine role of autonomous driving system in multi-car collision"},
                    {"title": "Tesla Stock Volatility Concerns Investors", "description": "Share price swings of 20% in single trading sessions raise questions about market stability"},
                    {"title": "Tesla Workers Report Safety Violations", "description": "Labor union documents unsafe working conditions at California manufacturing facility"},
                    {"title": "Tesla Delays Cybertruck Production Again", "description": "Electric pickup truck launch pushed back to 2024, disappointing pre-order customers"},
                ],
                'neutral': [
                    {"title": "Tesla Opens Manufacturing Plant in Germany", "description": "Gigafactory Berlin begins production with capacity for 500,000 vehicles annually"},
                    {"title": "Tesla Announces Battery Technology Partnership", "description": "Collaboration with Panasonic aims to improve energy density and reduce costs"},
                    {"title": "Tesla CEO Discusses Mars Mission Timeline", "description": "Elon Musk outlines SpaceX plans for human Mars colonization by 2029"},
                    {"title": "Tesla Expands Service Center Network", "description": "Company adds 200 new service locations to support growing customer base"},
                    {"title": "Tesla Participates in Clean Energy Summit", "description": "Company representatives discuss sustainable transportation at international conference"},
                ]
            },
            'climate change': {
                'positive': [
                    {"title": "Breakthrough in Carbon Capture Technology", "description": "New method removes CO2 from atmosphere 10x more efficiently, offering hope for climate goals"},
                    {"title": "Renewable Energy Costs Hit Record Lows", "description": "Solar and wind power now cheaper than fossil fuels in most markets worldwide"},
                    {"title": "Reforestation Project Exceeds Goals", "description": "International initiative plants 2 billion trees, surpassing targets by 150%"},
                    {"title": "Ocean Cleanup Technology Shows Success", "description": "Innovative system removes 100,000 tons of plastic waste from Pacific Ocean"},
                    {"title": "Electric Vehicle Sales Surge Globally", "description": "EV adoption reaches 40% of new car sales, accelerating transition from fossil fuels"},
                ],
                'negative': [
                    {"title": "Arctic Ice Melting Faster Than Predicted", "description": "Satellite data shows Greenland ice sheet losing mass at unprecedented rate"},
                    {"title": "Extreme Weather Events Increase Globally", "description": "Record-breaking hurricanes, floods, and wildfires attributed to climate change"},
                    {"title": "Coral Reefs Face Mass Bleaching Event", "description": "Rising ocean temperatures threaten 70% of world's coral ecosystems"},
                    {"title": "Climate Refugees Reach 100 Million", "description": "Displaced populations due to drought, flooding, and extreme weather continue growing"},
                    {"title": "Amazon Rainforest Deforestation Accelerates", "description": "Satellite imagery reveals 15% increase in forest loss over past year"},
                ],
                'neutral': [
                    {"title": "UN Climate Summit Convenes World Leaders", "description": "Representatives from 195 countries gather to discuss emission reduction strategies"},
                    {"title": "Climate Research Funding Increases", "description": "Governments allocate $50 billion for climate science and adaptation projects"},
                    {"title": "Green Technology Patents Rise 40%", "description": "Innovation in clean energy and sustainable technology reaches new heights"},
                    {"title": "Climate Education Programs Launch Globally", "description": "Schools worldwide integrate climate science into mandatory curricula"},
                    {"title": "Carbon Trading Markets Expand", "description": "New international frameworks for carbon credit exchange take effect"},
                ]
            }
        }
        
        # Generic templates for unknown queries
        self.generic_templates = {
            'positive': [
                {"title": "{query} Shows Promising Growth Trends", "description": "Recent market analysis indicates positive momentum in {query} sector with strong investor confidence"},
                {"title": "Breakthrough Innovation in {query} Industry", "description": "Revolutionary developments in {query} technology promise significant benefits for consumers"},
                {"title": "{query} Adoption Reaches New Milestones", "description": "Widespread acceptance of {query} solutions demonstrates market maturity and success"},
                {"title": "Investment in {query} Reaches Record Highs", "description": "Venture capital and institutional funding for {query} startups exceeds previous years"},
                {"title": "{query} Technology Wins Industry Awards", "description": "Leading {query} innovations recognized for excellence and positive impact"},
            ],
            'negative': [
                {"title": "Challenges Mount for {query} Industry", "description": "Market analysts identify significant obstacles facing {query} sector development"},
                {"title": "Regulatory Concerns Surround {query}", "description": "Government agencies raise questions about {query} safety and compliance issues"},
                {"title": "{query} Market Faces Volatility", "description": "Unpredictable market conditions create uncertainty for {query} investors and companies"},
                {"title": "Competition Intensifies in {query} Space", "description": "New entrants challenge established {query} market leaders with aggressive strategies"},
                {"title": "{query} Security Vulnerabilities Exposed", "description": "Cybersecurity experts identify potential risks in {query} systems and platforms"},
            ],
            'neutral': [
                {"title": "{query} Conference Brings Industry Together", "description": "Annual {query} summit features presentations from leading experts and thought leaders"},
                {"title": "New Research Published on {query}", "description": "Academic institutions release comprehensive studies on {query} trends and implications"},
                {"title": "{query} Market Analysis Report Released", "description": "Independent research firm provides detailed analysis of {query} industry dynamics"},
                {"title": "Standards Committee Reviews {query} Guidelines", "description": "Industry body updates best practices and recommendations for {query} implementation"},
                {"title": "{query} Partnership Announced", "description": "Major corporations collaborate on {query} initiatives to advance industry goals"},
            ]
        }

    def generate_articles(self, query, count=50):
        """Generate realistic sample articles for any query"""
        query_lower = query.lower()
        articles = []
        
        # Check if we have specific data for this query
        query_data = None
        for key in self.sample_data.keys():
            if key in query_lower or any(word in query_lower for word in key.split()):
                query_data = self.sample_data[key]
                break
        
        # Generate articles with realistic distribution (more neutral, balanced pos/neg)
        sentiments = ['positive'] * 15 + ['negative'] * 12 + ['neutral'] * 23  # 15:12:23 ratio
        random.shuffle(sentiments)
        
        sources = [
            'Reuters', 'Associated Press', 'BBC News', 'CNN', 'The Guardian', 'The New York Times',
            'The Wall Street Journal', 'Forbes', 'TechCrunch', 'Bloomberg', 'The Washington Post',
            'NPR', 'USA Today', 'CBS News', 'ABC News', 'NBC News', 'TIME', 'Newsweek',
            'Financial Times', 'The Economist'
        ]
        
        for i in range(count):
            sentiment = sentiments[i % len(sentiments)]
            
            if query_data and sentiment in query_data:
                # Use specific data
                template = random.choice(query_data[sentiment])
                title = template['title']
                description = template['description']
            else:
                # Use generic template
                template = random.choice(self.generic_templates[sentiment])
                title = template['title'].format(query=query.title())
                description = template['description'].format(query=query.lower())
            
            # Add variety to titles
            if i > 0:
                if i % 10 == 0:
                    title = f"Analysis: {title}"
                elif i % 15 == 0:
                    title = f"Breaking: {title}"
                elif i % 7 == 0:
                    title = f"Update: {title}"
            
            # Generate realistic article data
            publish_date = datetime.now() - timedelta(days=random.randint(1, 30))
            
            article = {
                'title': title,
                'description': description,
                'url': f"https://demo-news.com/article-{random.randint(1000, 9999)}",
                'urlToImage': f"https://picsum.photos/400/200?random={i}",
                'publishedAt': publish_date.isoformat(),
                'source': {'name': random.choice(sources)},
                'author': f"Reporter {random.randint(1, 100)}",
                'content': f"{description} [+{random.randint(500, 2000)} chars]",
                'relevance_score': random.randint(20, 50)
            }
            
            articles.append(article)
        
        return articles

# Global instance
demo_generator = DemoDataGenerator()
