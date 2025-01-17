import os
import streamlit as st
import wikipediaapi
import wikipedia
import requests
import spacy
import re
from dotenv import load_dotenv
from gtts import gTTS
import base64
import io
from groq import Groq
import time
import urllib.parse
import random

# Add custom CSS for layout and styling
st.markdown("""
<style>
    /* Header styling */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Title container */
    h1:first-of-type {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 300;
        letter-spacing: -0.5px;
        color: rgba(255, 255, 255, 0.9);
        font-size: 2.8rem !important;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    /* Subtitle styling */
    .stApp > div:first-child > div:nth-child(2) p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        margin-bottom: 3rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* Chat message icons */
    .stChatMessage [data-testid="stChatMessageAvatar"] {
        background: transparent !important;
        padding: 0.5rem;
    }

    /* User icon */
    .stChatMessage.user [data-testid="stChatMessageAvatar"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%) !important;
        border-radius: 12px;
    }

    /* Assistant icon */
    .stChatMessage.assistant [data-testid="stChatMessageAvatar"] {
        background: linear-gradient(135deg, #FFB86C 0%, #FFD93D 100%) !important;
        border-radius: 12px;
    }

    /* Chat message container */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        transition: all 0.2s ease-in-out;
    }

    .stChatMessage:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Main content width */
    .stApp > div:first-child {
        max-width: 1200px !important;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Base text style */
    .stChatMessage div.stMarkdown {
        color: rgba(250, 250, 250, 0.6) !important;
        line-height: 1.6;
        max-width: 100% !important;
    }
    
    /* Make chat messages wider */
    .stChatMessage {
        max-width: 100% !important;
    }
    
    .stChatMessage > div {
        max-width: 100% !important;
    }

    /* Link styling */
    .stChatMessage div.stMarkdown a {
        color: inherit !important;
        text-decoration: none !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }

    /* Importance-based text styling */
    .stChatMessage div.stMarkdown a[data-importance="primary"] {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500;
    }
    
    .stChatMessage div.stMarkdown a[data-importance="secondary"] {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    .stChatMessage div.stMarkdown a[data-importance="tertiary"] {
        color: rgba(255, 255, 255, 0.75) !important;
    }

    /* Subtle hover effect for links */
    .stChatMessage div.stMarkdown a:hover {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }
    
    /* Style the buttons container */
    div[data-testid="column"] > div {
        display: flex;
        justify-content: center;
        margin-top: 1.5rem;
    }
    
    /* Style the buttons */
    div[data-testid="column"] button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.75);
        transition: all 0.2s ease-in-out;
        min-height: unset;
        padding: 0.5rem 1rem;
        width: auto !important;
        flex: 1;
        border-radius: 8px;
    }
    
    div[data-testid="column"] button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Play button styling */
    .play-button-container {
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        z-index: 100;
    }
    
    .play-button {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        padding: 0;
    }
    
    .play-button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .play-button svg {
        width: 16px;
        height: 16px;
        fill: rgba(255, 255, 255, 0.8);
    }
    
    /* Make chat messages have relative positioning for play button */
    .stChatMessage {
        position: relative !important;
    }
    
    /* Styling for the collapsible sources section */
    details {
        transition: all 0.2s ease-in-out;
    }
    
    details summary {
        list-style: none;
        display: flex;
        align-items: center;
    }
    
    details summary::-webkit-details-marker {
        display: none;
    }
    
    details summary::before {
        content: "▶";
        margin-right: 0.5rem;
        transition: transform 0.2s ease-in-out;
        font-size: 0.8em;
        color: rgba(255, 255, 255, 0.6);
    }
    
    details[open] summary::before {
        transform: rotate(90deg);
    }
    
    details summary:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Scrollbar styling for sources content */
    details > div {
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.2) rgba(255, 255, 255, 0.05);
    }
    
    details > div::-webkit-scrollbar {
        width: 8px;
    }
    
    details > div::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    details > div::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
    
    details > div::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
</style>

<script>
function askFollowUp(question) {
    const input = document.querySelector('[data-testid="stChatInput"] input');
    const button = document.querySelector('[data-testid="stChatInput"] button');
    if (input && button) {
        input.value = question;
        button.click();
    }
}
</script>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia('HistoricalChatBot/1.0 (your@email.com)',
                            'en',
                            extract_format=wikipediaapi.ExtractFormat.WIKI)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except:
        # Download if not available
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

def create_wiki_link(text, importance):
    """Create a Wikipedia search link with proper formatting."""
    # Clean up the text and create a proper search URL
    clean_text = text.strip()
    search_url = f"https://en.wikipedia.org/wiki/Special:Search/{urllib.parse.quote(clean_text)}"
    
    # Return properly formatted HTML with data-importance attribute
    return f'<a href="{search_url}" class="wiki-link" data-importance="{importance}">{clean_text}</a>'

def add_wiki_links(text):
    """Process text and add Wikipedia links with importance-based styling."""
    # Clean up any existing URLs or markdown links
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Split text into sentences
    sentences = text.split('. ')
    result = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        words = sentence.split()
        i = 0
        phrase_result = []
        
        while i < len(words):
            # Try 3-word phrases (highest importance)
            if i + 2 < len(words):
                phrase = ' '.join(words[i:i+3])
                if (len(phrase) > 5 and 
                    not any(word.lower() in phrase.lower() for word in ['the', 'and', 'or', 'but']) and
                    any(word[0].isupper() for word in words[i:i+3])):
                    phrase_result.append(create_wiki_link(phrase, 'important'))
                    i += 3
                    continue
            
            # Try 2-word phrases (secondary importance)
            if i + 1 < len(words):
                phrase = ' '.join(words[i:i+2])
                if (len(phrase) > 5 and 
                    not any(word.lower() in phrase.lower() for word in ['the', 'and', 'or', 'but']) and
                    any(word[0].isupper() for word in words[i:i+2])):
                    phrase_result.append(create_wiki_link(phrase, 'secondary'))
                    i += 2
                    continue
            
            # Single words
            if words[i][0].isupper() and len(words[i]) > 2:
                phrase_result.append(create_wiki_link(words[i], 'important' if i == 0 else 'secondary'))
            else:
                phrase_result.append(words[i])
            i += 1
        
        result.append(' '.join(phrase_result))
    
    final_text = '. '.join(result).strip()
    return f'<div>{final_text}</div>'

def validate_wiki_content(text, title):
    """Validate that the Wikipedia content is relevant to the query."""
    try:
        # Extract key terms from the content
        doc = nlp(text[:1000])  # Limit to first 1000 chars for performance
        content_entities = set([ent.text.lower() for ent in doc.ents])
        
        # Calculate relevance scores
        term_matches = sum(term.lower() in text.lower() for term in key_terms)
        entity_overlap = sum(1 for term in key_terms if any(term.lower() in entity for entity in content_entities))
        
        # More lenient relevance criteria - just check if there's any meaningful overlap
        is_relevant = term_matches > 0 or entity_overlap > 0
        
        return is_relevant
        
    except Exception as e:
        return True  # Accept content if validation fails

def process_article(title, wiki_content, found_articles):
    """Process a Wikipedia article and add it to found_articles if relevant."""
    try:
        # Get article content
        page = wikipedia.page(title, auto_suggest=False)
        
        # Get summary and validate
        summary = page.summary
        if validate_wiki_content(summary, title):
            # Initialize article data
            found_articles[title] = {
                'summary': summary,
                'used_sections': []
            }
            
            # Add summary to wiki_content with relevance score
            wiki_content.append((summary, 5))  # Summary gets base score of 5
            
            # Process sections
            for section in page.sections:
                try:
                    section_content = page.section(section)
                    if section_content and validate_wiki_content(section_content, section):
                        found_articles[title]['used_sections'].append({
                            'name': section,
                            'content': section_content
                        })
                        wiki_content.append((section_content, 3))  # Sections get base score of 3
                except:
                    continue
                    
    except Exception as e:
        pass  # Skip articles that can't be processed

def get_wikipedia_content(query):
    """Search Wikipedia and get content for the query."""
    try:
        # Create a placeholder for showing search progress
        search_progress = st.empty()
        
        # Clean up query and extract key terms
        search_progress.markdown("🔍 Analyzing query and extracting key terms...")
        doc = nlp(query)
        
        # Extract dates and years
        years = set()
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                year_match = re.search(r'\b1[789]\d{2}\b|\b20\d{2}\b', ent.text)
                if year_match:
                    years.add(year_match.group(0))
        
        # Build search terms with historical context
        global key_terms
        key_terms = [ent.text for ent in doc.ents] + [token.text for token in doc if token.pos_ in ['PROPN', 'NOUN']]
        search_progress.markdown("📚 Building search strategy...")
        
        # Generate search terms
        search_terms = []
        
        # If we have a year, add year-based searches
        if years:
            for year in years:
                search_terms.extend([
                    f"{year} in history",
                    f"historical events {year}",
                    f"{year} history"
                ])
                # Combine year with key terms
                for term in key_terms:
                    search_terms.append(f"{term} {year}")
        
        # Add general search terms
        search_terms.extend([
            query + " historical event",
            query + " in history",
            query
        ] + key_terms)
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        # Process main search results
        wiki_content = []
        seen_content = set()
        found_articles = {}
        total_content_found = 0
        
        for term in search_terms:
            try:
                search_progress.markdown(f"🔍 Searching for: {term}")
                search_results = wikipedia.search(term, results=8)
                for title in search_results:
                    if title not in found_articles:  # Only process new articles
                        process_article(title, wiki_content, found_articles)
                        if title in found_articles:  # If article was added
                            total_content_found += 1
                    
                if total_content_found >= 3 and any(score >= 5 for _, score in wiki_content):
                    break
                    
            except Exception as e:
                continue
        
        # Sort content by relevance
        wiki_content.sort(key=lambda x: x[1], reverse=True)
        
        # Show final search completion message
        search_progress.markdown(f"""✅ Search complete
Found content from {total_content_found} sources""")
        time.sleep(2)
        search_progress.empty()
        
        if wiki_content:
            st.session_state['last_wiki_articles'] = found_articles
            return "\n\n".join(content for content, _ in wiki_content[:8])
        return None
    except Exception as e:
        st.error(f"Error searching Wikipedia: {str(e)}")
        return None

def get_deepseek_response(prompt, wiki_content):
    """Get response from Deepseek API with follow-up suggestions."""
    try:
        # First try to get API key from session state
        api_key = st.session_state.get('DEEPSEEK_API_KEY')
        if not api_key:
            # If not in session state, try to get from secrets
            try:
                api_key = st.secrets["DEEPSEEK_API_KEY"]
            except:
                return "Please enter your Deepseek API key in the sidebar to continue."
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build conversation history context
        conversation_context = ""
        if 'messages' in st.session_state and len(st.session_state.messages) > 0:
            recent_messages = st.session_state.messages[-6:]
            conversation_context = "\nPrevious conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = re.sub(r'<[^>]+>', '', msg["content"])
                content = re.sub(r'\[(\d)\]\[([^\]]+)\]', r'\2', content)
                conversation_context += f"{role}: {content}\n"
        
        # Combine wiki content with user's question and conversation context
        full_prompt = f"""You are a knowledgeable historical chatbot. Your task is to provide a detailed response using ONLY the Wikipedia content provided below. Do not include any information that is not from these sources.

Wikipedia Content:
{wiki_content}

Previous Conversation:
{conversation_context}

Current Question: {prompt}

REQUIREMENTS:
1. Use ONLY information from the provided Wikipedia content
2. Mark important terms using these exact markers:
   - Use [1][term] for major historical figures, key events, and primary concepts
   - Use [2][term] for dates, places, and technical terms
   - Use [3][term] for supporting concepts and contextual details
3. For each fact or claim in your response, mentally note which Wikipedia article or section it came from
4. End your response with exactly three follow-up questions, each on a new line starting with [SUGGESTION]

Keep your response natural and flowing, without section headers or numbering. Focus on creating a clear hierarchy of information through your term marking."""

        system_prompt = """You are a knowledgeable historical chatbot that provides detailed responses using ONLY the Wikipedia content provided. Never include information from outside the provided sources. Mark important concepts with:
- [1][text] for major historical developments, inventions, and key figures
- [2][text] for specific dates, places, and technical terms
- [3][text] for additional context and details (use sparingly)

Keep your response natural and flowing. Do not overuse importance markers - only mark truly significant terms. Never use phrases like 'supporting detail:' in your response."""

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        try:
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'deepseek-chat',
                    'messages': messages
                }
            )
            response_text = response.json()['choices'][0]['message']['content']
            
            # Split response and suggestions
            parts = response_text.split('\n')
            main_response = []
            suggestions = []
            
            for part in parts:
                if part.strip().startswith('[SUGGESTION]'):
                    suggestions.append(part.replace('[SUGGESTION]', '').strip())
                else:
                    main_response.append(part)
            
            main_response = ' '.join(main_response).strip()
            
            # Store exactly 3 suggestions in session state
            if 'suggestions' not in st.session_state:
                st.session_state.suggestions = []
            st.session_state.suggestions = suggestions[:3]
            
            # If we don't have enough suggestions, generate some based on found articles
            if 'last_wiki_articles' in st.session_state and len(st.session_state.suggestions) < 3:
                article_titles = list(st.session_state['last_wiki_articles'].keys())
                while len(st.session_state.suggestions) < 3 and article_titles:
                    title = random.choice(article_titles)
                    st.session_state.suggestions.append(f"Tell me more about {title}")
                    article_titles.remove(title)
            
            # If still not enough suggestions, add generic ones
            while len(st.session_state.suggestions) < 3:
                st.session_state.suggestions.append("What other historical inventions were significant during this period?")
            
            # Clean up formatting artifacts
            main_response = re.sub(r'PART \d:', '', main_response)
            main_response = re.sub(r'\*\*.*?\*\*', '', main_response)
            main_response = re.sub(r'\d\. ', '', main_response)
            main_response = re.sub(r'Follow-Up Questions:', '', main_response)
            
            # Process URLs
            main_response = re.sub(r'https?://\S+', '', main_response)
            main_response = re.sub(r'\(https?://[^)]+\)', '', main_response)
            
            # Process importance markers in main response
            for level in range(1, 4):
                main_response = re.sub(
                    f'\\[{level}\\]\\[([^\\]]+)\\]',
                    lambda m: create_wiki_link(m.group(1), 
                        'important' if level == 1 else 'secondary' if level == 2 else 'tertiary'),
                    main_response
                )
            
            # Remove any remaining instances of the word "term"
            main_response = re.sub(r'\bterm\b', '', main_response)
            
            # Clean up extra spaces and normalize whitespace
            main_response = re.sub(r'\s+', ' ', main_response)
            main_response = main_response.strip()
            
            return f'<div>{main_response}</div>'
        except Exception as e:
            return f"Error communicating with Deepseek API: {str(e)}"
    except Exception as e:
        return f"Error communicating with Deepseek API: {str(e)}"

def get_groq_response(prompt, wiki_content):
    """Get response from Groq API with follow-up suggestions."""
    try:
        api_key = st.session_state.get('GROQ_API_KEY')
        if not api_key:
            return "Please enter your Groq API key in the sidebar to continue. You can get a free key from groq.com"
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Build conversation history context
        conversation_context = ""
        if 'messages' in st.session_state and len(st.session_state.messages) > 0:
            recent_messages = st.session_state.messages[-6:]
            conversation_context = "\nPrevious conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = re.sub(r'<[^>]+>', '', msg["content"])
                content = re.sub(r'\[(\d)\]\[([^\]]+)\]', r'\2', content)
                conversation_context += f"{role}: {content}\n"
        
        # Combine wiki content with user's question and conversation context
        full_prompt = f"""You are a knowledgeable historical chatbot. Your task is to provide a detailed response using ONLY the Wikipedia content provided below. Do not include any information that is not from these sources.

Wikipedia Content:
{wiki_content}

Previous Conversation:
{conversation_context}

Current Question: {prompt}

REQUIREMENTS:
1. Use ONLY information from the provided Wikipedia content
2. Mark important terms using these exact markers:
   - Use [1][term] for major historical figures, key events, and primary concepts
   - Use [2][term] for dates, places, and technical terms
   - Use [3][term] for supporting concepts and contextual details
3. For each fact or claim in your response, mentally note which Wikipedia article or section it came from
4. End your response with exactly three follow-up questions, each on a new line starting with [SUGGESTION]

Keep your response natural and flowing, without section headers or numbering. Focus on creating a clear hierarchy of information through your term marking."""

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": """You are a knowledgeable historical chatbot that provides detailed responses using ONLY the Wikipedia content provided. Never include information from outside the provided sources. Mark important terms with:
- [1][term] for major figures and primary concepts
- [2][term] for dates, places, and technical terms
- [3][term] for supporting details"""
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.7,
            max_tokens=4096,
            top_p=1
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Split response and suggestions
        parts = response_text.split('\n')
        main_response = []
        suggestions = []
        
        for part in parts:
            if part.strip().startswith('[SUGGESTION]'):
                suggestions.append(part.replace('[SUGGESTION]', '').strip())
            else:
                main_response.append(part)
        
        main_response = ' '.join(main_response).strip()
        
        # Store exactly 3 suggestions in session state
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = []
        st.session_state.suggestions = suggestions[:3]
        
        # If we don't have enough suggestions, generate some based on found articles
        if 'last_wiki_articles' in st.session_state and len(st.session_state.suggestions) < 3:
            article_titles = list(st.session_state['last_wiki_articles'].keys())
            while len(st.session_state.suggestions) < 3 and article_titles:
                title = random.choice(article_titles)
                st.session_state.suggestions.append(f"Tell me more about {title}")
                article_titles.remove(title)
        
        # If still not enough suggestions, add generic ones
        while len(st.session_state.suggestions) < 3:
            st.session_state.suggestions.append("What other historical inventions were significant during this period?")
        
        # Clean up formatting artifacts
        main_response = re.sub(r'PART \d:', '', main_response)
        main_response = re.sub(r'\*\*.*?\*\*', '', main_response)
        main_response = re.sub(r'\d\. ', '', main_response)
        main_response = re.sub(r'Follow-Up Questions:', '', main_response)
        
        # Process URLs
        main_response = re.sub(r'https?://\S+', '', main_response)
        main_response = re.sub(r'\(https?://[^)]+\)', '', main_response)
        
        # Process importance markers in main response
        for level in range(1, 4):
            main_response = re.sub(
                f'\\[{level}\\]\\[([^\\]]+)\\]',
                lambda m: create_wiki_link(m.group(1), 
                    'important' if level == 1 else 'secondary' if level == 2 else 'tertiary'),
                main_response
            )
        
        # Remove any remaining instances of the word "term"
        main_response = re.sub(r'\bterm\b', '', main_response)
        
        # Clean up extra spaces and normalize whitespace
        main_response = re.sub(r'\s+', ' ', main_response)
        main_response = main_response.strip()
        
        return f'<div>{main_response}</div>'
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def validate_content(prompt, response_text):
    """Validate that the AI response stays on topic and relevant to the prompt."""
    try:
        # Extract key terms from the prompt
        prompt_doc = nlp(prompt)
        prompt_entities = set([ent.text.lower() for ent in prompt_doc.ents])
        prompt_nouns = set([token.text.lower() for token in prompt_doc if token.pos_ in ['PROPN', 'NOUN']])
        prompt_key_terms = prompt_entities.union(prompt_nouns)
        
        # Add historical context terms
        historical_terms = {'history', 'century', 'period', 'era', 'empire', 'kingdom', 'state', 'nation', 'population', 'people'}
        prompt_key_terms.update(historical_terms)
        
        # Extract key terms from the response
        response_doc = nlp(response_text)
        response_entities = set([ent.text.lower() for ent in response_doc.ents])
        
        # Check if the response contains entities not related to the prompt
        unrelated_entities = []
        for entity in response_entities:
            # Skip common words, short terms, and historical context terms
            if (len(entity) < 4 or 
                entity.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'} or
                entity.lower() in historical_terms):
                continue
                
            # Check if this entity is related to any prompt terms
            is_related = False
            for prompt_term in prompt_key_terms:
                if (prompt_term in entity.lower() or 
                    entity.lower() in prompt_term or 
                    prompt_term.split()[-1] in entity.lower() or
                    entity.lower().split()[-1] in prompt_term):
                    is_related = True
                    break
            
            if not is_related:
                unrelated_entities.append(entity)
        
        # Only flag as invalid if there are multiple unrelated entities
        if len(unrelated_entities) > 3:
            correction_prompt = f"""Your previous response included too many unrelated topics: {', '.join(unrelated_entities[:3])}...
            
Please provide a new response that focuses ONLY on {prompt} without mentioning unrelated people, events, or concepts.
Use the same Wikipedia content but stay strictly focused on the topic."""
            
            return False, correction_prompt
            
        return True, None
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return True, None  # Continue with original response if validation fails

def get_ai_response(prompt, wiki_content):
    """Get response from selected AI model with follow-up suggestions."""
    try:
        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Get initial response
        if model_choice == "Deepseek (Requires API Key)":
            response = get_deepseek_response(prompt, wiki_content)
        else:
            response = get_groq_response(prompt, wiki_content)
        
        # Extract text content from HTML response
        clean_response = re.sub(r'<[^>]+>', '', response)
        clean_response = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', clean_response)
        
        # Validate content
        is_valid, correction_prompt = validate_content(prompt, clean_response)
        
        # If content is not valid, get a new response
        if not is_valid and correction_prompt:
            if model_choice == "Deepseek (Requires API Key)":
                response = get_deepseek_response(correction_prompt, wiki_content)
            else:
                response = get_groq_response(correction_prompt, wiki_content)
            
        # Add sources to the response if we have them
        if 'last_wiki_articles' in st.session_state:
            sources_html = '<div style="margin-top: 2rem; margin-bottom: 1rem;">'
            sources_html += '<details style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;">'
            sources_html += '<summary style="padding: 1rem; cursor: pointer; user-select: none; font-weight: 500; color: rgba(255, 255, 255, 0.8);">Content Used from Wikipedia</summary>'
            sources_html += '<div style="max-height: 300px; overflow-y: auto; padding: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">'
            
            for title, data in st.session_state['last_wiki_articles'].items():
                # Only show articles that contributed content
                if data['summary'] or data['used_sections']:
                    sources_html += f'<div style="margin-bottom: 1.5rem;">'
                    # Article title with link
                    sources_html += f'<div style="margin-bottom: 0.5rem;">'
                    sources_html += f'<a href="https://en.wikipedia.org/wiki/{title.replace(" ", "_")}" target="_blank" style="color: rgba(255, 255, 255, 0.8); text-decoration: none; border-bottom: 1px dotted rgba(255, 255, 255, 0.3); font-weight: 500;">{title}</a>'
                    sources_html += '</div>'
                    
                    # Main summary
                    if data['summary']:
                        sources_html += '<div style="margin-left: 1rem; margin-bottom: 0.5rem;">'
                        sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.3rem;"><em>From main article:</em></div>'
                        sources_html += f'<div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9em; line-height: 1.4;">{data["summary"][:200]}...</div>'
                        sources_html += '</div>'
                    
                    # Used sections
                    if data['used_sections']:
                        sources_html += '<div style="margin-left: 1rem;">'
                        sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.3rem;"><em>Additional sections used:</em></div>'
                        for section in data['used_sections']:
                            sources_html += f'<div style="margin-bottom: 0.5rem;">'
                            sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.2rem;">• {section["name"]}</div>'
                            sources_html += f'<div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9em; margin-left: 0.5rem; line-height: 1.4;">{section["content"][:150]}...</div>'
                            sources_html += '</div>'
                        sources_html += '</div>'
                    
                    sources_html += '</div>'
            
            sources_html += '</div></details></div>'
            
            # Add sources to the response while preserving any existing HTML
            if response.endswith('</div>'):
                response = response[:-6] + sources_html + '</div>'
            else:
                response += sources_html
                
        return response
            
    except Exception as e:
        return f"Error communicating with AI model: {str(e)}"

# Initialize session state for wiki references if not exists
if 'wiki_references' not in st.session_state:
    st.session_state.wiki_references = []

# Main content area
st.markdown('<h1>Historical Chat Bot</h1>', unsafe_allow_html=True)
st.markdown('<p>Ask me anything about history! I\'ll combine Wikipedia knowledge with AI insights.</p>', unsafe_allow_html=True)

# Chat history and input
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

# Sidebar with setup and model selection
with st.sidebar:
    st.title("About")
    st.write("""
    This Historical Chat Bot combines information from Wikipedia with AI-powered insights 
    to provide comprehensive answers to your historical questions.
    
    Ask any question about history, and I'll provide detailed answers with clickable 
    Wikipedia links for key people, places, events, and concepts.
    """)
    
    st.title("Setup")
    st.session_state['model_choice'] = st.selectbox(
        "Choose AI Model:",
        ["Groq (Free)", "Deepseek (Requires API Key)"],
        help="Groq is free to use. Deepseek requires your own API key."
    )
    
    if st.session_state['model_choice'] == "Deepseek (Requires API Key)":
        api_key = st.text_input("Enter your Deepseek API key:", type="password", help="Your API key will only be stored for this session")
        if api_key:
            st.session_state['DEEPSEEK_API_KEY'] = api_key
            st.success("Deepseek API key saved for this session!")
    else:
        api_key = st.text_input("Enter your Groq API key:", type="password", help="Get a free API key from groq.com")
        if api_key:
            st.session_state['GROQ_API_KEY'] = api_key
            st.success("Groq API key saved for this session!")

def get_audio_base64(text):
    """Generate audio from text and return as base64."""
    try:
        # Remove HTML tags and clean up text
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', clean_text)
        
        # Create an in-memory bytes buffer
        mp3_fp = io.BytesIO()
        
        # Generate audio using gTTS (British accent)
        tts = gTTS(text=clean_text, lang='en', tld='co.uk')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(mp3_fp.read()).decode()
        return audio_base64
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Create columns for play button and message
            cols = st.columns([0.9, 0.1])
            with cols[1]:
                if st.button("🔊", key=f"play_{idx}", help="Play audio"):
                    # Generate and play audio
                    clean_text = re.sub(r'<[^>]+>', '', message['content'])
                    clean_text = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', clean_text)
                    
                    # Create audio in memory
                    mp3_fp = io.BytesIO()
                    tts = gTTS(text=clean_text, lang='en', tld='co.uk')
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)
                    
                    # Play audio using HTML5 audio
                    audio_bytes = mp3_fp.read()
                    st.audio(audio_bytes, format='audio/mp3')
            
            with cols[0]:
                st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"], unsafe_allow_html=True)
        
        # Show suggestion buttons only for the most recent assistant message
        if (message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            st.session_state.suggestions):
            cols = st.columns(len(st.session_state.suggestions))
            for i, (col, suggestion) in enumerate(zip(cols, st.session_state.suggestions)):
                # Clean up the suggestion text
                clean_suggestion = suggestion.strip()
                # Remove importance markers and clean up text
                clean_suggestion = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', clean_suggestion)
                clean_suggestion = re.sub(r'\s+', ' ', clean_suggestion)
                # Use a unique key combining message index and suggestion index
                button_key = f"suggestion_{idx}_{i}"
                if col.button(clean_suggestion, key=button_key):
                    st.session_state.messages.append({"role": "user", "content": clean_suggestion})
                    wiki_content = get_wikipedia_content(clean_suggestion)
                    if wiki_content:
                        response = get_ai_response(clean_suggestion, wiki_content)
                    else:
                        response = get_ai_response(clean_suggestion, "No direct Wikipedia article found for this query.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

# Chat input
if prompt := st.chat_input("What would you like to know about history?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    wiki_content = get_wikipedia_content(prompt)
    
    if wiki_content:
        response = get_ai_response(prompt, wiki_content)
    else:
        response = get_ai_response(prompt, "No direct Wikipedia article found for this query.")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun() 