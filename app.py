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

def create_wiki_link(text, importance='supporting'):
    """Create a Wikipedia link for the given text with importance-based styling."""
    # Clean up text
    clean_text = text.strip()
    
    # Common words to skip
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as',
        'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will',
        'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
        'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year',
        'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its',
        'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
        'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'was', 'were', 'had', 'has', 'been',
        'said', 'did', 'many', 'more', 'those', 'is', 'am', 'are', 'very', 'much'
    }
    
    # Skip if text is too short, just numbers, or common words
    if (len(clean_text) < 3 or 
        clean_text.isdigit() or 
        clean_text.lower() in common_words or 
        len(clean_text.split()) == 1 and clean_text.lower() in common_words):
        return text
    
    # Create a search URL
    search_url = f"https://en.wikipedia.org/w/index.php?search={clean_text.replace(' ', '+')}"
    return f'<a href="{search_url}" data-importance="{importance}">{text}</a>'

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
        key_terms = [ent.text for ent in doc.ents] + [token.text for token in doc if token.pos_ in ['PROPN', 'NOUN']]
        search_progress.markdown("📚 Building search strategy...")
        
        # Track search metrics
        total_articles_found = 0
        relevant_articles_found = 0
        max_search_time = 30  # Maximum search time in seconds
        start_time = time.time()
        
        # Generate broader search terms
        search_terms = []
        
        # Economic/industry specific terms for better context
        industry_terms = [
            "economic history",
            "industrial history",
            "industry statistics",
            "economic statistics",
            "industrial production",
            "economic growth",
            "major industries",
            "industrial development",
            "economic sectors",
            "business history"
        ]
        
        # If we have a year, prioritize searches with that year
        if years:
            for year in years:
                # Add year-specific search terms
                search_terms.extend([
                    f"{year} in history",
                    f"historical events {year}",
                    f"{year} history",
                    f"economy in {year}",
                    f"industrial history {year}",
                    f"economic conditions {year}",
                ])
                # Combine year with other key terms
                for term in key_terms:
                    search_terms.append(f"{term} {year}")
                # Add industry-specific year combinations
                for term in industry_terms:
                    search_terms.append(f"{term} {year}")
        
        # Add general historical search terms
        search_terms.extend([
            query + " historical event",
            query + " in history",
            query + " economic history",
            query + " industrial history",
            query
        ] + key_terms + industry_terms)
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        wiki_content = []
        seen_content = set()
        found_articles = {}  # Track which articles contributed what content
        processed_count = 0
        
        def should_continue_search():
            """Determine if the search should continue based on various metrics."""
            current_time = time.time()
            search_duration = current_time - start_time
            
            # Stop conditions:
            # 1. Exceeded max search time
            if search_duration > max_search_time:
                return False
                
            # 2. Found enough high-quality articles
            if len(found_articles) >= 20 and relevant_articles_found >= 10:
                return False
                
            # 3. Processed too many articles with diminishing returns
            if processed_count > 200:
                return False
                
            return True
        
        def process_article(title, depth=0, max_depth=2):
            """Process an article and its related links up to max_depth."""
            nonlocal processed_count, relevant_articles_found, key_terms
            
            if not should_continue_search() or depth > max_depth or title in found_articles:
                return
            
            try:
                processed_count += 1
                search_progress.markdown(
                    f"🔍 Searching... (Articles processed: {processed_count})\n\n"
                    f"Currently analyzing: {title}\n\n"
                    f"Relevant articles found: {relevant_articles_found}"
                )
                
                page = wiki.page(title)
                if not page.exists():
                    return
                    
                # Get summary and check relevance
                summary = page.summary
                
                # Validate summary content
                if not validate_wiki_content(summary, title, key_terms):
                    return
                    
                # Track what content we're using from this article
                found_articles[title] = {
                    'summary': summary[:1000],
                    'used_sections': [],
                    'relevance': 'primary' if depth == 0 else 'related'
                }
                
                relevant_articles_found += 1
                
                # Add summary if unique
                if summary not in seen_content:
                    seen_content.add(summary)
                    wiki_content.append(f"From article '{title}':\n{summary[:1000]}")
                    
                # Look for relevant sections
                sections = page.sections
                if sections and should_continue_search():
                    relevant_sections = []
                    for section in sections:
                        section_text = page.section_by_title(section)
                        if len(section_text) > 100 and validate_wiki_content(section_text, title, key_terms):
                            relevant_sections.append((section, section_text))
                    
                    # Sort sections by relevance
                    relevant_sections.sort(key=lambda x: sum(term.lower() in x[1].lower() for term in key_terms), reverse=True)
                    
                    # Take top 5 most relevant sections
                    for section, section_text in relevant_sections[:5]:
                        if not should_continue_search():
                            break
                        section_summary = section_text[:500]
                        if section_summary not in seen_content:
                            seen_content.add(section_summary)
                            wiki_content.append(f"Additional context from section '{section}':\n{section_summary}")
                            found_articles[title]['used_sections'].append({
                                'name': section,
                                'content': section_summary
                            })
                
                # If this is not too deep, follow links to related articles
                if depth < max_depth and should_continue_search():
                    # Get links from the page
                    links = page.links
                    # Sort links by relevance to our key terms
                    relevant_links = []
                    for link in links:
                        relevance_score = sum(term.lower() in link.lower() for term in key_terms)
                        if relevance_score > 0:
                            relevant_links.append((link, relevance_score))
                    
                    # Sort by relevance score and process top 5 related articles
                    relevant_links.sort(key=lambda x: x[1], reverse=True)
                    for link, _ in relevant_links[:5]:
                        if not should_continue_search():
                            break
                        process_article(link, depth + 1, max_depth)
                        
            except Exception as e:
                return
        
        # Process main search results
        for term in search_terms:
            if not should_continue_search():
                break
                
            try:
                search_progress.markdown(f"🔍 Searching for: {term}")
                search_results = wikipedia.search(term, results=5)
                for title in search_results:
                    if not should_continue_search():
                        break
                    process_article(title)
                    
            except Exception as e:
                continue
        
        search_duration = time.time() - start_time
        search_progress.markdown(
            f"✅ Search complete!\n\n"
            f"• Found {len(found_articles)} articles ({relevant_articles_found} relevant)\n"
            f"• Processed {processed_count} total articles\n"
            f"• Search took {search_duration:.1f} seconds"
        )
        time.sleep(2)  # Show completion message briefly
        search_progress.empty()  # Clear the progress display
        
        if wiki_content:
            # Store found articles in session state for reference
            st.session_state['last_wiki_articles'] = found_articles
            return "\n\n".join(wiki_content)
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

        try:
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {
                            'role': 'system',
                            'content': """You are a knowledgeable historical chatbot that provides detailed responses using ONLY the Wikipedia content provided. Never include information from outside the provided sources. Mark important terms with:
- [1][term] for major figures and primary concepts
- [2][term] for dates, places, and technical terms
- [3][term] for supporting details"""
                        },
                        {
                            'role': 'user',
                            'content': full_prompt
                        }
                    ]
                }
            )
            response_text = response.json()['choices'][0]['message']['content']
            
            # Split response and suggestions
            parts = response_text.split('[SUGGESTION]')
            main_response = parts[0].strip()
            suggestions = [s.strip() for s in parts[1:] if s.strip()]
            
            # Clean up formatting artifacts
            main_response = re.sub(r'PART \d:', '', main_response)
            main_response = re.sub(r'\*\*.*?\*\*', '', main_response)
            main_response = re.sub(r'\d\. ', '', main_response)
            main_response = re.sub(r'Follow-Up Questions:', '', main_response)
            
            # Process the main response
            main_response = re.sub(r'https?://\S+', '', main_response)
            main_response = re.sub(r'\(https?://[^)]+\)', '', main_response)
            
            # Process importance markers in main response
            for level in range(1, 4):
                main_response = re.sub(
                    f'\\[{level}\\]\\[([^\\]]+)\\]',
                    lambda m: create_wiki_link(m.group(1), 
                        'primary' if level == 1 else 'secondary' if level == 2 else 'tertiary'),
                    main_response
                )
            
            # Clean up extra spaces and normalize whitespace
            main_response = re.sub(r'\s+', ' ', main_response)
            main_response = main_response.strip()
            
            # Store suggestions in session state
            if 'suggestions' not in st.session_state:
                st.session_state.suggestions = []
            st.session_state.suggestions = [s.strip() for s in suggestions[:3]]
            
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
        parts = response_text.split('[SUGGESTION]')
        main_response = parts[0].strip()
        suggestions = [s.strip() for s in parts[1:] if s.strip()]
        
        # Clean up formatting artifacts
        main_response = re.sub(r'PART \d:', '', main_response)
        main_response = re.sub(r'\*\*.*?\*\*', '', main_response)
        main_response = re.sub(r'\d\. ', '', main_response)
        main_response = re.sub(r'Follow-Up Questions:', '', main_response)
        
        # Process the main response
        main_response = re.sub(r'https?://\S+', '', main_response)
        main_response = re.sub(r'\(https?://[^)]+\)', '', main_response)
        
        # Process importance markers in main response
        for level in range(1, 4):
            main_response = re.sub(
                f'\\[{level}\\]\\[([^\\]]+)\\]',
                lambda m: create_wiki_link(m.group(1), 
                    'primary' if level == 1 else 'secondary' if level == 2 else 'tertiary'),
                main_response
            )
        
        # Clean up extra spaces and normalize whitespace
        main_response = re.sub(r'\s+', ' ', main_response)
        main_response = main_response.strip()
        
        # Store suggestions in session state
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = []
        st.session_state.suggestions = [s.strip() for s in suggestions[:3]]
        
        return f'<div>{main_response}</div>'
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def validate_wiki_content(text, title, key_terms=None):
    """Validate that the Wikipedia content is relevant to the query."""
    try:
        # If no key terms provided, extract them from the text
        if key_terms is None:
            doc = nlp(text[:1000])
            key_terms = [ent.text for ent in doc.ents] + [token.text for token in doc if token.pos_ in ['PROPN', 'NOUN']]
        
        # Extract key terms from the content
        doc = nlp(text[:1000])  # Limit to first 1000 chars for performance
        content_entities = set([ent.text.lower() for ent in doc.ents])
        
        # Extract dates and years
        years = set()
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                # Try to extract year from date entity
                year_match = re.search(r'\b1[789]\d{2}\b|\b20\d{2}\b', ent.text)
                if year_match:
                    years.add(year_match.group(0))
        
        # If the title contains a year, it must be relevant to our key terms
        title_year_match = re.search(r'\b1[789]\d{2}\b|\b20\d{2}\b', title)
        if title_year_match:
            title_year = title_year_match.group(0)
            # Check if this year is relevant to our query
            year_relevance = any(term.lower() in text.lower() for term in key_terms)
            if not year_relevance:
                return False
        
        # Calculate relevance scores
        term_matches = sum(term.lower() in text.lower() for term in key_terms)
        entity_overlap = sum(1 for term in key_terms if any(term.lower() in entity for entity in content_entities))
        
        # Check for historical context words
        historical_terms = {'history', 'historical', 'event', 'period', 'era', 'century', 'decade', 'war', 'revolution', 'movement', 'reign', 'rule', 'dynasty', 'empire', 'kingdom', 'government', 'politics', 'society', 'culture', 'economy'}
        historical_context = sum(1 for term in historical_terms if term in text.lower())
        
        # Content must meet ANY of these criteria:
        # 1. Have key term matches
        # 2. Have entity overlap
        # 3. Have historical context if it's not a primary source
        # 4. If it contains a year that matches our query
        
        # Check for scientific/astronomical terms that might indicate irrelevant content
        scientific_terms = {'galaxy', 'cluster', 'constellation', 'star', 'planet', 'physics', 'quantum', 'chemical', 'molecule'}
        has_scientific_terms = any(term in text.lower() for term in scientific_terms)
        
        # Calculate final relevance - more lenient criteria
        is_relevant = (
            (term_matches > 0 or  # Has any term matches
            entity_overlap > 0 or  # Has any entity overlap
            historical_context > 0) and  # Has any historical context
            not has_scientific_terms and  # Must not be scientific/astronomical
            (not title_year_match or year_relevance)  # If it has a year, it must be relevant
        )
        
        return is_relevant
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return True  # Accept content if validation fails

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
        
        # Always add sources to the response if we have them
        if 'last_wiki_articles' in st.session_state and st.session_state['last_wiki_articles']:
            sources_html = '<div style="margin-top: 2rem; margin-bottom: 1rem;">'
            sources_html += '<details style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;">'
            sources_html += '<summary style="padding: 1rem; cursor: pointer; user-select: none; font-weight: 500; color: rgba(255, 255, 255, 0.8);">Content Used from Wikipedia</summary>'
            sources_html += '<div style="max-height: 300px; overflow-y: auto; padding: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">'
            
            for title, data in st.session_state['last_wiki_articles'].items():
                sources_html += f'<div style="margin-bottom: 1.5rem;">'
                # Article title with link
                sources_html += f'<div style="margin-bottom: 0.5rem;">'
                sources_html += f'<a href="https://en.wikipedia.org/wiki/{title.replace(" ", "_")}" target="_blank" style="color: rgba(255, 255, 255, 0.8); text-decoration: none; border-bottom: 1px dotted rgba(255, 255, 255, 0.3); font-weight: 500;">{title}</a>'
                sources_html += '</div>'
                
                # Main summary
                if data.get('summary'):
                    sources_html += '<div style="margin-left: 1rem; margin-bottom: 0.5rem;">'
                    sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.3rem;"><em>From main article:</em></div>'
                    sources_html += f'<div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9em; line-height: 1.4;">{data["summary"][:200]}...</div>'
                    sources_html += '</div>'
                
                # Used sections
                if data.get('used_sections'):
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
            if '</div>' in response:
                response = response.replace('</div>', sources_html + '</div>', 1)
            else:
                response = f'<div>{response}{sources_html}</div>'
        
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