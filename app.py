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

def process_text_importance(text):
    """Process text and add visual hierarchy using NLP."""
    if nlp is None:
        return text  # Return unprocessed text if NLP isn't available
        
    # First clean up any formatting artifacts and ensure proper spacing
    text = re.sub(r'(\d+)\s*-\s*(st|nd|rd|th)', r'\1\2', text)  # Remove spaces in ordinals with dashes
    text = re.sub(r'(\d+)\s+(st|nd|rd|th)', r'\1\2', text)  # Remove spaces in ordinals
    
    # Add spaces between numbers and letters (except for ordinals)
    text = re.sub(r'(\d+)(?!(st|nd|rd|th))([A-Za-z])', r'\1 \2', text)  # Add space between numbers and letters
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Add space between letters and numbers
    
    # Remove any existing markdown formatting
    text = re.sub(r'\*\*\s*\*\*', '', text)  # Remove empty bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove existing bold markers
    
    try:
        doc = nlp(text)
        
        # Track processed terms to avoid duplicates
        processed_terms = set()
        
        # First pass: Named entities and important historical concepts
        for ent in doc.ents:
            if ent.text not in processed_terms and len(ent.text.strip()) > 0:
                if ent.label_ in ['DATE', 'TIME', 'EVENT']:
                    # Use HTML span instead of markdown for dates
                    text = re.sub(rf'\b{re.escape(ent.text)}\b', f'<span class="primary-term">{ent.text}</span>', text)
                elif ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG']:
                    text = re.sub(rf'\b{re.escape(ent.text)}\b', f'<span class="secondary-term">{ent.text}</span>', text)
                processed_terms.add(ent.text)
        
        # Second pass: Important historical concepts and terms
        historical_concepts = [
            'American imperialism', 'indigenous communities', 'cultural erasure',
            'self-determination', 'political representation', 'traditional ways',
            'ancestral territories', 'power imbalances', 'cultural autonomy',
            'discriminatory policies', 'political participation'
        ]
        
        for concept in historical_concepts:
            if concept.lower() in text.lower() and concept not in processed_terms:
                text = re.sub(rf'\b{re.escape(concept)}\b', f'<span class="tertiary-term">{concept}</span>', text, flags=re.IGNORECASE)
                processed_terms.add(concept)
        
        # Clean up any empty markup that might have been generated
        text = re.sub(r'<span[^>]*>\s*</span>', '', text)  # Remove empty spans
        
        return f'<div>{text}</div>'
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return text  # Return unprocessed text if there's an error

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
            # Ensure messages are in the correct order (oldest to newest)
            messages = conv.messages or []
            st.session_state.messages = messages
            st.session_state.current_conversation_id = conv_id
            st.session_state.suggestions = []
            # Clear any existing suggestions since we're loading a new conversation
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

# Add custom CSS for layout and styling
st.markdown("""
<style>
    /* Base text style */
    .stChatMessage div.stMarkdown {
        line-height: 1.8;
        font-size: 1.1rem;
        max-width: 80ch;
        margin: 0 auto;
    }
    
    /* Primary terms (bold) */
    .stChatMessage div.stMarkdown strong {
        color: rgba(255, 255, 255, 1);
        font-weight: 500;
        font-size: inherit;
    }
    
    /* Secondary terms (emphasis) */
    .stChatMessage div.stMarkdown em {
        color: rgba(255, 255, 255, 0.75);
        font-style: normal;
        font-size: inherit;
    }
    
    /* Tertiary terms (code) */
    .stChatMessage div.stMarkdown code {
        color: rgba(255, 255, 255, 0.55);
        font-style: normal;
        background: none;
        font-family: inherit;
        font-size: inherit;
        padding: 0;
    }
    
    /* Chat message container */
    .stChatMessage {
        padding: 2rem;
        margin: 1.5rem 0;
        background: rgba(0, 0, 0, 0.2);
    }
    
    /* Paragraph spacing */
    .stChatMessage .stMarkdown p {
        margin-bottom: 1.5rem;
    }
    
    /* Follow-up questions section */
    .stChatMessage hr {
        margin: 2rem 0;
        opacity: 0.2;
    }
    
    /* Buttons styling */
    div[data-testid="column"] button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1.25rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    div[data-testid="column"] button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia('HistoricalChatBot/1.0 (your@email.com)',
                            'en',
                            extract_format=wikipediaapi.ExtractFormat.WIKI)

# Load spaCy model
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """Load spaCy model with error handling and download if needed."""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        # If model isn't found, download it
        spacy.cli.download('en_core_web_sm')
        return spacy.load('en_core_web_sm')
    except Exception as e:
        st.error(f"Error loading spaCy model: {str(e)}")
        return None

# Initialize NLP
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
            try:
                api_key = st.secrets["DEEPSEEK_API_KEY"]
            except:
                return "Please enter your Deepseek API key in the sidebar to continue."
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build conversation history context only if there are previous messages
        conversation_context = ""
        if 'messages' in st.session_state and len(st.session_state.messages) > 1:
            recent_messages = st.session_state.messages[-6:-1]
            if recent_messages:
                conversation_context = "\nPrevious conversation:\n"
                for msg in recent_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = re.sub(r'<[^>]+>', '', msg["content"])
                    content = re.sub(r'\[(\d)\]\[([^\]]+)\]', r'\2', content)
                    conversation_context += f"{role}: {content}\n"
        
        # Combine wiki content with user's question and conversation context
        full_prompt = f"""Context from Wikipedia: {wiki_content}
{conversation_context}Current Question: {prompt}

You are a knowledgeable historical expert. Provide a coherent, well-structured response that:

1. Introduces the main topic clearly with specific dates
2. Develops ideas in a logical sequence
3. Uses clear transitions between related concepts
4. Explains cause-and-effect relationships
5. Provides proper context for all historical terms and events

Critical requirements:
- ALWAYS begin with a complete date in the first sentence
- NEVER omit years when referencing dates
- ALWAYS include the full year when concluding
- Maintain specific dates throughout the entire response
- Double-check that no dates are missing before completing the response

Important guidelines:
- Start with a clear introduction of the main topic
- Develop one idea fully before moving to the next
- Use clear transition phrases between ideas
- Explain terms and concepts before using them
- Keep sentences focused and avoid run-on structures
- End with a clear conclusion that includes the specific year

Then provide three follow-up questions that probe deeper into different aspects of the topic. Format each as a complete question without colons. Start each with [SUGGESTION].

Keep the response natural and flowing, without section headers or numbering."""

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
            
            # Clean up extra spaces and normalize whitespace
            main_response = re.sub(r'\s+', ' ', main_response)
            main_response = main_response.strip()
            
            # Store suggestions in session state
            if 'suggestions' not in st.session_state:
                st.session_state.suggestions = []
            st.session_state.suggestions = [s.strip() for s in suggestions[:3]]
            
            return main_response
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
        
        # Build conversation history context only if there are previous messages
        conversation_context = ""
        if 'messages' in st.session_state and len(st.session_state.messages) > 1:
            recent_messages = st.session_state.messages[-6:-1]
            if recent_messages:
                conversation_context = "\nPrevious conversation:\n"
                for msg in recent_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = re.sub(r'<[^>]+>', '', msg["content"])
                    content = re.sub(r'\[(\d)\]\[([^\]]+)\]', r'\2', content)
                    conversation_context += f"{role}: {content}\n"
        
        # Combine wiki content with user's question and conversation context
        full_prompt = f"""Context from Wikipedia: {wiki_content}
{conversation_context}Current Question: {prompt}

You are a knowledgeable historical expert. Provide a coherent, well-structured response that:

1. Introduces the main topic clearly with specific dates
2. Develops ideas in a logical sequence
3. Uses clear transitions between related concepts
4. Explains cause-and-effect relationships
5. Provides proper context for all historical terms and events

Critical requirements:
- ALWAYS begin with a complete date in the first sentence
- NEVER omit years when referencing dates
- ALWAYS include the full year when concluding
- Maintain specific dates throughout the entire response
- Double-check that no dates are missing before completing the response

Important guidelines:
- Start with a clear introduction of the main topic
- Develop one idea fully before moving to the next
- Use clear transition phrases between ideas
- Explain terms and concepts before using them
- Keep sentences focused and avoid run-on structures
- End with a clear conclusion that includes the specific year

Then provide three follow-up questions that probe deeper into different aspects of the topic. Format each as a complete question without colons. Start each with [SUGGESTION].

Keep the response natural and flowing, without section headers or numbering."""
        
        # Get raw response from API
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable historical expert who provides clear, coherent responses with accurate dates and proper marking of important terms."
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
        
        # Get raw response text
        response_text = completion.choices[0].message.content.strip()
        
        # Split response and suggestions
        parts = response_text.split('[SUGGESTION]')
        main_response = parts[0].strip()
        suggestions = [s.strip() for s in parts[1:] if s.strip()]
        
        # Store suggestions in session state
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = []
        st.session_state.suggestions = [s.strip() for s in suggestions[:3]]
        
        return main_response
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def get_ai_response(prompt, wiki_content):
    """Get response from selected AI model."""
    try:
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Get raw response from selected model
        if model_choice == "Deepseek (Requires API Key)":
            response = get_deepseek_response(prompt, wiki_content)
        else:
            response = get_groq_response(prompt, wiki_content)
        
        # Split response and suggestions
        parts = response.split('[SUGGESTION]')
        main_response = parts[0].strip()
        
        # Store suggestions
        if len(parts) > 1:
            suggestions = [s.strip() for s in parts[1:] if s.strip()]
            st.session_state.suggestions = suggestions[:3]
        
        # Process the response with NLP-based importance first
        processed_response = process_text_importance(main_response)
        
        # Then clean up any remaining formatting artifacts
        processed_response = re.sub(r'PART \d:', '', processed_response)
        processed_response = re.sub(r'\*\*.*?\*\*', '', processed_response)
        processed_response = re.sub(r'\d\. ', '', processed_response)
        processed_response = re.sub(r'Follow-Up Questions:', '', processed_response)
        processed_response = re.sub(r'https?://\S+', '', processed_response)
        processed_response = re.sub(r'\(https?://[^)]+\)', '', processed_response)
        processed_response = re.sub(r'\s+', ' ', processed_response)
        
        return processed_response.strip()
        
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
                # Get the most recent message as the title, or use a default
                title = "Empty Conversation"
                if conv.messages and len(conv.messages) > 0:
                    last_msg = conv.messages[-1]["content"]  # Get last message instead of first
                    # Clean up the title text
                    last_msg = re.sub(r'<[^>]+>', '', last_msg)  # Remove HTML tags
                    last_msg = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', last_msg)  # Remove importance markers
                    title = last_msg[:50] + "..." if len(last_msg) > 50 else last_msg
                
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
            cols = st.columns([0.8, 0.1, 0.1])
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
            
            with cols[2]:
                if idx == len(st.session_state.messages) - 1:  # Only for latest message
                    if st.button("🎲", key=f"shuffle_{idx}", help="Generate new response and suggestions"):
                        # Get the last user message for context
                        last_user_msg = None
                        for msg in reversed(st.session_state.messages[:-1]):  # Exclude current message
                            if msg["role"] == "user":
                                last_user_msg = msg["content"]
                                break
                        
                        if last_user_msg:
                            wiki_content = get_wikipedia_content(last_user_msg)
                            if not wiki_content:
                                wiki_content = "No direct Wikipedia article found for this query."
                            
                            # Get new response with new suggestions
                            response = get_ai_response(last_user_msg, wiki_content)
                            # Update the current message content
                            st.session_state.messages[-1]["content"] = response
                            save_conversation()
                            st.rerun()
            
            with cols[0]:
                st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"], unsafe_allow_html=True)
        
        # Show suggestion buttons only for the most recent assistant message
        if (message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            st.session_state.suggestions):
            
            st.markdown("---")  # Add a separator
            st.write("Follow-up Questions:")
            
            cols = st.columns(len(st.session_state.suggestions))
            for i, (col, suggestion) in enumerate(zip(cols, st.session_state.suggestions)):
                # Clean up the suggestion text
                clean_suggestion = suggestion.strip()
                # Remove all formatting artifacts
                clean_suggestion = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', clean_suggestion)
                clean_suggestion = re.sub(r'\[.*?\]', '', clean_suggestion)
                clean_suggestion = re.sub(r'\d+\.\s*', '', clean_suggestion)
                clean_suggestion = re.sub(r'[{}]', '', clean_suggestion)
                clean_suggestion = re.sub(r'\s*-\s*$', '', clean_suggestion)  # Remove trailing dashes
                clean_suggestion = re.sub(r'\s+-\s+', ' ', clean_suggestion)  # Remove dashes between words
                clean_suggestion = re.sub(r'^\s*-\s*', '', clean_suggestion)  # Remove leading dashes
                clean_suggestion = re.sub(r'\s+', ' ', clean_suggestion)
                clean_suggestion = clean_suggestion.strip()
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