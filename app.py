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
from models import User, Conversation, get_db_session
from datetime import datetime

# Initialize session states
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'wiki_references' not in st.session_state:
    st.session_state.wiki_references = []

def save_api_keys():
    """Save API keys to user's profile."""
    if st.session_state.user_id:
        db = get_db_session()
        try:
            user = db.query(User).get(st.session_state.user_id)
            if user:
                user.deepseek_api_key = st.session_state.get('DEEPSEEK_API_KEY')
                user.groq_api_key = st.session_state.get('GROQ_API_KEY')
                db.commit()
        finally:
            db.close()

def get_user_conversations():
    """Get all conversations for the current user."""
    if not st.session_state.user_id:
        return []
    db = get_db_session()
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == st.session_state.user_id
        ).order_by(Conversation.updated_at.desc()).all()
        return conversations or []
    finally:
        db.close()

def load_conversation(conv_id):
    """Load a specific conversation."""
    db = get_db_session()
    try:
        conv = db.query(Conversation).get(conv_id)
        if conv and conv.user_id == st.session_state.user_id:
            st.session_state.messages = conv.messages or []
            st.session_state.current_conversation_id = conv_id
            st.session_state.suggestions = []
    finally:
        db.close()

def create_new_conversation():
    """Create a new conversation."""
    if st.session_state.messages:
        save_conversation()
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.current_conversation_id = None

def save_conversation():
    """Save current conversation to database."""
    if st.session_state.user_id and st.session_state.messages:
        db = get_db_session()
        try:
            if st.session_state.current_conversation_id:
                conv = db.query(Conversation).get(st.session_state.current_conversation_id)
                if conv:
                    conv.messages = st.session_state.messages
                    conv.updated_at = datetime.utcnow()
                    db.commit()
                    return
            
            # Create new conversation
            conv = Conversation(
                user_id=st.session_state.user_id,
                messages=st.session_state.messages
            )
            db.add(conv)
            db.commit()
            st.session_state.current_conversation_id = conv.id
        finally:
            db.close()

def login_user(username, password):
    """Login user and return success status."""
    db = get_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.commit()
            st.session_state.user_id = user.id
            st.session_state.username = user.username
            # Load user's API keys
            if user.deepseek_api_key:
                st.session_state['DEEPSEEK_API_KEY'] = user.deepseek_api_key
            if user.groq_api_key:
                st.session_state['GROQ_API_KEY'] = user.groq_api_key
            # Load last conversation
            last_conv = db.query(Conversation).filter(
                Conversation.user_id == user.id
            ).order_by(Conversation.updated_at.desc()).first()
            if last_conv:
                st.session_state.messages = last_conv.messages or []
                st.session_state.current_conversation_id = last_conv.id
            return True
        return False
    finally:
        db.close()

def signup_user(username, password):
    """Create new user account and return success status."""
    db = get_db_session()
    try:
        if db.query(User).filter(User.username == username).first():
            return False
        user = User(username=username)
        user.set_password(password)
        db.add(user)
        db.commit()
        st.session_state.user_id = user.id
        st.session_state.username = user.username
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        return True
    finally:
        db.close()

def logout_user():
    """Logout user and clear session state."""
    if st.session_state.messages:
        save_conversation()
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.current_conversation_id = None
    if 'DEEPSEEK_API_KEY' in st.session_state:
        del st.session_state['DEEPSEEK_API_KEY']
    if 'GROQ_API_KEY' in st.session_state:
        del st.session_state['GROQ_API_KEY']

def delete_conversation(conv_id):
    """Delete a specific conversation."""
    if not st.session_state.user_id:
        return False
    db = get_db_session()
    try:
        conv = db.query(Conversation).get(conv_id)
        if conv and conv.user_id == st.session_state.user_id:
            # If we're deleting the current conversation, clear the state
            if st.session_state.current_conversation_id == conv_id:
                st.session_state.messages = []
                st.session_state.suggestions = []
                st.session_state.current_conversation_id = None
            db.delete(conv)
            db.commit()
            return True
        return False
    finally:
        db.close()

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
        # Search for the query
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return None
            
        wiki_content = []
        
        # Get content for each result
        for title in search_results:
            try:
                page = wiki.page(title)
                if page.exists():
                    wiki_content.append(page.summary[0:500])
            except Exception as e:
                continue
        
        if wiki_content:
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
            # Get last few exchanges, but limit to keep context manageable
            recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges (3 pairs of messages)
            conversation_context = "\nPrevious conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                # Clean up any HTML/markdown from previous responses
                content = re.sub(r'<[^>]+>', '', msg["content"])
                content = re.sub(r'\[(\d)\]\[([^\]]+)\]', r'\2', content)
                conversation_context += f"{role}: {content}\n"
        
        # Combine wiki content with user's question and conversation context
        full_prompt = f"""Context from Wikipedia: {wiki_content}
{conversation_context}
Current Question: {prompt}

Respond in two parts:

PART 1: Provide a detailed response about the topic that takes into account the previous conversation context when relevant. Mark important terms using these markers:
- [1][term] for major historical figures, key events, primary concepts
- [2][term] for dates, places, technical terms
- [3][term] for related concepts and supporting details

PART 2: Provide three follow-up questions that build upon both the current topic and previous context, each on a new line starting with [SUGGESTION]. Make the questions natural and conversational.

Keep the response natural and flowing, without section headers or numbering. Mark only the most relevant terms, and ensure they're marked exactly once."""

        try:
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json={
                    'model': 'deepseek-chat',
                    'messages': [{'role': 'user', 'content': full_prompt}]
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
        full_prompt = f"""You are a knowledgeable historical chatbot. Your task is to provide a detailed response about the following topic, with careful attention to marking important terms.

Context from Wikipedia: {wiki_content}
{conversation_context}
Current Question: {prompt}

CRITICAL FORMATTING REQUIREMENTS:
1. You MUST mark ALL important terms using these exact markers:
   - Use [1][term] for major historical figures (e.g., [1][Julius Caesar]), key events (e.g., [1][Battle of Waterloo]), and primary concepts
   - Use [2][term] for dates (e.g., [2][44 BC]), places (e.g., [2][Roman Empire]), and technical terms
   - Use [3][term] for supporting concepts and contextual details

2. Example of properly marked text:
"[1][Napoleon Bonaparte] led the [1][French Army] into [2][Russia] in [2][1812], employing [3][scorched earth tactics] during the campaign."

3. Follow these marking rules strictly:
   - Mark EVERY important term - aim for at least 2-3 terms per sentence
   - Use [1] for the most important 25% of terms
   - Use [2] for the next 50% of terms
   - Use [3] for the remaining 25% of terms
   - Never mark the same term twice
   - Always include the full term in the brackets

4. End your response with exactly three follow-up questions, each on a new line starting with [SUGGESTION]

Keep your response natural and flowing, without section headers or numbering. Focus on creating a clear hierarchy of information through your term marking."""

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": """You are a knowledgeable historical chatbot. Your primary task is to mark important terms with the correct importance level:
- [1][term] for major figures and primary concepts (25% of terms)
- [2][term] for dates, places, and technical terms (50% of terms)
- [3][term] for supporting details (25% of terms)
Mark EVERY important term, and ensure proper hierarchy through your marking."""
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

def get_ai_response(prompt, wiki_content):
    """Get response from selected AI model with follow-up suggestions."""
    try:
        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        if model_choice == "Deepseek (Requires API Key)":
            return get_deepseek_response(prompt, wiki_content)
        else:
            return get_groq_response(prompt, wiki_content)
    except Exception as e:
        return f"Error communicating with AI model: {str(e)}"

# Main content area
st.markdown('<h1>Historical Chat Bot</h1>', unsafe_allow_html=True)
st.markdown('<p>Ask me anything about history! I\'ll combine Wikipedia knowledge with AI insights.</p>', unsafe_allow_html=True)

# Sidebar with setup and model selection
with st.sidebar:
    if not st.session_state.user_id:
        st.title("Login / Sign Up")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login_user(login_username, login_password):
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            signup_username = st.text_input("Choose Username", key="signup_username")
            signup_password = st.text_input("Choose Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            if st.button("Sign Up"):
                if signup_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif signup_user(signup_username, signup_password):
                    st.success("Account created successfully!")
                    st.rerun()
                else:
                    st.error("Username already exists")
    else:
        st.title(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            logout_user()
            st.rerun()
        
        # Conversation Management
        st.title("Conversations")
        if st.button("Start New Conversation", type="primary"):
            create_new_conversation()
            st.rerun()
        
        # List previous conversations
        conversations = get_user_conversations()
        if conversations:
            st.write("Previous Conversations:")
            for conv in conversations:
                # Get the first message as the title, or use a default
                title = "Empty Conversation"
                if conv.messages and len(conv.messages) > 0:
                    first_msg = conv.messages[0]["content"]
                    title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
                
                col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
                with col1:
                    if st.button(f"📝 {title}", key=f"conv_{conv.id}"):
                        load_conversation(conv.id)
                        st.rerun()
                with col2:
                    st.write(conv.updated_at.strftime("%Y-%m-%d"))
                with col3:
                    if st.button("🗑️", key=f"del_{conv.id}", help="Delete conversation"):
                        if delete_conversation(conv.id):
                            st.success("Conversation deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete conversation")
        
        st.title("Setup")
        st.session_state['model_choice'] = st.selectbox(
            "Choose AI Model:",
            ["Groq (Free)", "Deepseek (Requires API Key)"],
            help="Groq is free to use. Deepseek requires your own API key."
        )
        
        if st.session_state['model_choice'] == "Deepseek (Requires API Key)":
            api_key = st.text_input(
                "Enter your Deepseek API key:", 
                type="password",
                value=st.session_state.get('DEEPSEEK_API_KEY', ''),
                help="Your API key will be saved to your account"
            )
            if api_key:
                st.session_state['DEEPSEEK_API_KEY'] = api_key
                save_api_keys()
                st.success("Deepseek API key saved!")
        else:
            api_key = st.text_input(
                "Enter your Groq API key:", 
                type="password",
                value=st.session_state.get('GROQ_API_KEY', ''),
                help="Get a free API key from groq.com"
            )
            if api_key:
                st.session_state['GROQ_API_KEY'] = api_key
                save_api_keys()
                st.success("Groq API key saved!")
        
        st.title("About")
        st.write("""
        This Historical Chat Bot combines information from Wikipedia with AI-powered insights 
        to provide comprehensive answers to your historical questions.
        
        Ask any question about history, and I'll provide detailed answers with clickable 
        Wikipedia links for key people, places, events, and concepts.
        """)

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
    if not st.session_state.user_id:
        st.error("Please login or sign up to start chatting")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    wiki_content = get_wikipedia_content(prompt)
    
    if wiki_content:
        response = get_ai_response(prompt, wiki_content)
    else:
        response = get_ai_response(prompt, "No direct Wikipedia article found for this query.")

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_conversation()  # Save after each message
    st.rerun() 