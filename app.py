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
import json
import concurrent.futures

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
    """Search Wikipedia and get content for the query using AI-driven search."""
    try:
        # Create a placeholder for showing search progress
        search_progress = st.empty()
        search_progress.markdown("🤔 Analyzing query to develop search strategy...")
        
        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Check for API key before proceeding
        if model_choice == "Deepseek (Requires API Key)":
            api_key = st.session_state.get('DEEPSEEK_API_KEY')
            if not api_key:
                st.error("Please enter your Deepseek API key in the sidebar to use AI-powered search.")
                return None
        else:  # Groq
            api_key = st.session_state.get('GROQ_API_KEY')
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar to use AI-powered search.")
                return None
        
        # Initialize default search terms as fallback
        default_search_terms = [
            query,  # Exact query
            query + " history",  # Historical context
            query.split(" in ")[0] if " in " in query else query,  # Main topic
            query.split(" in ")[1] if " in " in query else "",  # Time period/location
            query + " impact",  # Impact/significance
            query + " background",  # Historical background
            query + " effects",  # Effects/consequences
            query + " significance",  # Historical significance
            query + " period",  # Time period
            query + " era"  # Historical era
        ]
        default_search_terms = [term for term in default_search_terms if term]  # Remove empty terms
        
        # Try to get AI-generated search terms
        try:
            search_progress.markdown("🧠 Generating intelligent search strategy...")
            search_terms = get_ai_search_terms(query, model_choice)
            
            if not search_terms:
                search_progress.warning("AI search term generation failed, using fallback strategy...")
                search_terms = default_search_terms
            else:
                search_progress.success(f"Generated {len(search_terms)} search terms")
                
        except Exception as e:
            search_progress.warning(f"Error generating search terms: {str(e)}\nUsing fallback strategy...")
            search_terms = default_search_terms
        
        # Initialize tracking variables
        wiki_content = []
        seen_content = set()
        found_articles = {}
        processed_count = 0
        relevant_articles_found = 0
        last_progress_update = time.time()
        search_start_time = time.time()
        
        # Search progress tracking
        search_progress.markdown("🔍 Beginning search process...")
        
        # Process each search term
        for term_index, term in enumerate(search_terms):
            # Check overall search timeout (5 minutes)
            if time.time() - search_start_time > 300:
                search_progress.markdown("⚠️ Search timeout reached. Using available results...")
                break
                
            # Update progress less frequently to avoid UI lag
            current_time = time.time()
            if current_time - last_progress_update >= 0.5:
                search_progress.markdown(
                    f"🔍 Searching... (Articles processed: {processed_count})\n\n"
                    f"Currently trying: {term}\n\n"
                    f"Relevant articles found: {relevant_articles_found}\n\n"
                    f"Search term {term_index + 1} of {len(search_terms)}"
                )
                last_progress_update = current_time
            
            try:
                # Get initial search results with timeout
                search_results = wikipedia.search(term, results=20)  # Increased from 10 to 20
                term_start_time = time.time()
                
                # For each potential article
                for title in search_results:
                    # Check per-term timeout (60 seconds)
                    if time.time() - term_start_time > 60:  # Increased from 30 to 60
                        break
                        
                    processed_count += 1
                    
                    if title in found_articles:
                        continue
                        
                    try:
                        page = wiki.page(title)
                        if not page.exists():
                            continue
                            
                        # Get full content for validation
                        content_to_validate = f"Title: {title}\n\nSummary: {page.summary}\n\nContent: {page.content[:2000]}"
                        
                        # Full AI validation with timeout
                        try:
                            with st.spinner(f"Validating article: {title}"):
                                is_relevant = validate_article_with_timeout(page, query, model_choice)
                        except Exception as e:
                            continue
                            
                        if is_relevant:
                            relevant_articles_found += 1
                            
                            # Store article info
                            found_articles[title] = {
                                'summary': page.summary[:1000],
                                'used_sections': [],
                                'relevance': 0.8  # Default high relevance for validated articles
                            }
                            
                            # Add summary to content
                            if page.summary not in seen_content:
                                seen_content.add(page.summary)
                                wiki_content.append(f"From article '{title}':\n{page.summary[:1000]}")
                            
                            # If we have enough articles, we can stop
                            if relevant_articles_found >= 10:  # Increased from 5 to 10
                                break
                                
                    except Exception as e:
                        continue
                        
                # If we have enough articles, we can stop
                if relevant_articles_found >= 10:  # Increased from 5 to 10
                    break
                    
            except Exception as e:
                continue
            
            # If we've processed too many articles without finding any relevant ones,
            # continue to the next search term
            if processed_count >= 100 and relevant_articles_found == 0:  # Increased from 50 to 100
                continue
        
        # Show final stats
        search_progress.markdown(
            f"✅ Search complete!\n\n"
            f"• Found {len(found_articles)} relevant articles\n"
            f"• Processed {processed_count} total articles\n"
            f"• Search time: {time.time() - search_start_time:.1f} seconds"
        )
        time.sleep(2)
        search_progress.empty()
        
        if wiki_content and found_articles:
            st.session_state['last_wiki_articles'] = found_articles
            return "\n\n".join(wiki_content)
            
        if not wiki_content:
            st.error("No relevant articles found. Please try rephrasing your question.")
        return None
        
    except Exception as e:
        st.error(f"Error searching Wikipedia: {str(e)}")
        return None

def quick_relevance_check(text, query):
    """Quick check for basic relevance before full AI validation."""
    # Convert to lower case for comparison
    text = text.lower()
    query = query.lower()
    
    # Extract key terms from query
    query_terms = query.split()
    
    # Check if any query terms appear in the text
    term_matches = sum(1 for term in query_terms if term in text)
    
    # Extract years from query and text
    query_years = re.findall(r'\b1[0-9]{3}\b|\b20[0-9]{2}\b', query)
    text_years = re.findall(r'\b1[0-9]{3}\b|\b20[0-9]{2}\b', text)
    
    # If query has years, at least one should match
    if query_years and not any(year in text_years for year in query_years):
        return False
    
    # Return true if we have enough term matches
    return term_matches >= 2

def validate_article_with_timeout(page, query, model_choice, timeout=10):
    """Validate article relevance with a timeout."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(validate_wiki_content, page.summary, page.title, query.split())
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return False
    except Exception as e:
        return False

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

def validate_content(prompt, response_text):
    """Validate that the AI response stays on topic and relevant to the prompt."""
    try:
        # Extract key terms from the prompt
        prompt_doc = nlp(prompt)
        prompt_entities = set([ent.text.lower() for ent in prompt_doc.ents])
        prompt_nouns = set([token.text.lower() for token in prompt_doc if token.pos_ in ['PROPN', 'NOUN']])
        prompt_key_terms = prompt_entities.union(prompt_nouns)
        
        # Extract key terms from the response
        response_doc = nlp(response_text)
        response_entities = set([ent.text.lower() for ent in response_doc.ents])
        
        # Check if the response contains entities not related to the prompt
        unrelated_entities = []
        for entity in response_entities:
            # Skip common words and short terms
            if len(entity) < 4 or entity.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                continue
                
            # Check if this entity is related to any prompt terms
            is_related = False
            for prompt_term in prompt_key_terms:
                if (prompt_term in entity.lower() or 
                    entity.lower() in prompt_term or 
                    prompt_term.split()[-1] in entity.lower() or  # Check last word of multi-word terms
                    entity.lower().split()[-1] in prompt_term):  # Check last word of multi-word entities
                    is_related = True
                    break
            
            if not is_related:
                unrelated_entities.append(entity)
        
        if unrelated_entities:
            # Create a new prompt to get a more focused response
            correction_prompt = f"""Your previous response included unrelated topics: {', '.join(unrelated_entities)}
            
Please provide a new response that focuses ONLY on {prompt} without mentioning unrelated people, events, or concepts.
Use the same Wikipedia content but stay strictly focused on the topic."""
            
            return False, correction_prompt
            
        return True, None
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return True, None  # Continue with original response if validation fails

def validate_wiki_content(text, title, key_terms=None):
    """Validate that the Wikipedia content is relevant to the query using AI."""
    try:
        # Build validation prompt
        validation_prompt = f"""Analyze if this Wikipedia content is relevant to the search terms.

Search Terms: {', '.join(key_terms) if key_terms else 'None provided'}
Article Title: {title}

Content to validate:
{text[:2000]}

Requirements for Validation:
1. Content Analysis:
   - Does the content directly address the search terms?
   - Are key historical figures, events, or concepts from the search present?
   - Is the historical context relevant?

2. Depth Check:
   - Does the content provide substantial information?
   - Are there specific details, dates, or events mentioned?
   - Is it more than just a passing reference?

3. Context Validation:
   - Is the content historically focused?
   - Does it provide background or consequences?
   - Are there connections to broader historical themes?

4. Relevance Scoring:
   - How central is this content to the search topic? (0-100%)
   - Are there significant sections about the search topic?
   - Would this help answer questions about the topic?

Respond with a JSON object:
{
    "is_relevant": true/false,
    "relevance_score": 0-100,
    "key_matches": ["list", "of", "matched", "terms"],
    "reasoning": "Brief explanation of decision"
}"""

        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Get validation response
        if model_choice == "Deepseek (Requires API Key)":
            api_key = st.session_state.get('DEEPSEEK_API_KEY')
            if not api_key:
                return False
                
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'deepseek-chat',
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are a strict content validator. Respond only with the requested JSON format.'
                        },
                        {
                            'role': 'user',
                            'content': validation_prompt
                        }
                    ],
                    'temperature': 0.3
                }
            )
            result = json.loads(response.json()['choices'][0]['message']['content'].strip())
            
        else:
            api_key = st.session_state.get('GROQ_API_KEY')
            if not api_key:
                return False
                
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict content validator. Respond only with the requested JSON format."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500,
                top_p=1
            )
            
            result = json.loads(completion.choices[0].message.content.strip())
        
        # Article must be relevant AND have a high enough relevance score
        return result.get('is_relevant', False) and result.get('relevance_score', 0) >= 60
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return False  # Reject content if validation fails

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

def get_ai_search_terms(query, model_choice):
    """Generate intelligent search terms using AI."""
    try:
        st.write("Debug: Starting search term generation...")
        
        # Build search strategy prompt
        strategy_prompt = f"""As a historical research assistant, analyze this query and develop a comprehensive search strategy.

Query: {query}

Create a search strategy that will help find relevant Wikipedia articles. Consider:
1. Key historical figures, events, and concepts
2. Relevant time periods and locations
3. Related historical contexts and themes
4. Alternative names or terms that might be used
5. Broader historical context and related topics

Respond with ONLY a JSON array of search terms, ordered from most specific to most general. Example:
["term 1", "term 2", "term 3"]"""

        st.write(f"Debug: Using model: {model_choice}")
        
        # Get AI-generated search strategy
        if model_choice == "Deepseek (Requires API Key)":
            api_key = st.session_state.get('DEEPSEEK_API_KEY')
            if not api_key:
                st.write("Debug: No Deepseek API key found")
                return None
                
            try:
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                st.write("Debug: Sending request to Deepseek API...")
                response = requests.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers=headers,
                    json={
                        'model': 'deepseek-chat',
                        'messages': [
                            {
                                'role': 'system',
                                'content': 'You are a historical research assistant. Respond only with a JSON array of search terms.'
                            },
                            {
                                'role': 'user',
                                'content': strategy_prompt
                            }
                        ],
                        'temperature': 0.7
                    },
                    timeout=10
                )
                st.write("Debug: Got response from Deepseek API")
                st.write(f"Debug: Response status: {response.status_code}")
                st.write(f"Debug: Response content: {response.text[:200]}...")
                
                search_terms = json.loads(response.json()['choices'][0]['message']['content'])
                st.write(f"Debug: Parsed search terms: {search_terms[:3]}...")
                
            except Exception as e:
                st.write(f"Debug: Deepseek API error: {str(e)}")
                raise e
            
        else:
            api_key = st.session_state.get('GROQ_API_KEY')
            if not api_key:
                st.write("Debug: No Groq API key found")
                return None
                
            try:
                st.write("Debug: Initializing Groq client...")
                client = Groq(api_key=api_key)
                
                st.write("Debug: Sending request to Groq API...")
                completion = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a historical research assistant. Respond only with a JSON array of search terms."
                        },
                        {
                            "role": "user",
                            "content": strategy_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500,
                    top_p=1,
                    request_timeout=10
                )
                st.write("Debug: Got response from Groq API")
                st.write(f"Debug: Response content: {completion.choices[0].message.content[:200]}...")
                
                search_terms = json.loads(completion.choices[0].message.content)
                st.write(f"Debug: Parsed search terms: {search_terms[:3]}...")
                
            except Exception as e:
                st.write(f"Debug: Groq API error: {str(e)}")
                raise e
        
        # Validate and clean up search terms
        if not isinstance(search_terms, list):
            st.write("Debug: AI did not return a list")
            raise ValueError("AI did not return a list of search terms")
            
        # Remove any empty or non-string terms
        search_terms = [str(term).strip() for term in search_terms if term and isinstance(term, (str, int, float))]
        st.write(f"Debug: Cleaned terms count: {len(search_terms)}")
        
        # Remove duplicates while preserving order
        seen = set()
        search_terms = [x for x in search_terms if not (x.lower() in seen or seen.add(x.lower()))]
        st.write(f"Debug: Final terms count after deduplication: {len(search_terms)}")
        
        # Ensure we have at least some terms
        if not search_terms:
            st.write("Debug: No valid search terms after cleanup")
            raise ValueError("No valid search terms generated")
            
        return search_terms
        
    except Exception as e:
        st.write(f"Debug: Final error in get_ai_search_terms: {str(e)}")
        return None 