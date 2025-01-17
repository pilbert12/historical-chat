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

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Set page config
st.set_page_config(page_title="Historical Chat Bot", page_icon="📚")

# Add custom CSS
st.markdown("""
<style>
    /* Link styling */
    .stMarkdown a {
        color: inherit !important;
        text-decoration: none !important;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    
    /* Importance-based text styling */
    .stMarkdown a[data-importance="important"] {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500;
    }
    
    .stMarkdown a[data-importance="secondary"] {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    .stMarkdown a[data-importance="tertiary"] {
        color: rgba(255, 255, 255, 0.75) !important;
    }
    
    /* Subtle hover effect for links */
    .stMarkdown a:hover {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
        padding: 2px 4px;
        margin: -2px -4px;
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
    """Create a Wikipedia link with proper styling based on importance."""
    clean_text = text.strip()
    # Use full Wikipedia URL and properly encode the title
    wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(clean_text.replace(' ', '_'))}"
    # Use JavaScript window.open to handle the link click
    return f'<a href="{wiki_url}" data-importance="{importance}" target="_blank" onclick="window.open(this.href, \'_blank\'); return false;">{clean_text}</a>'

def add_wiki_links(text):
    """Process text and add Wikipedia links with importance-based styling."""
    # Clean up any existing URLs or markdown links
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Process importance markers with non-greedy matching and proper boundary handling
    text = re.sub(r'\[1\]\[([^\]]+?)\](?=\s|$|[.,!?])', lambda m: create_wiki_link(m.group(1), 'important'), text)
    text = re.sub(r'\[2\]\[([^\]]+?)\](?=\s|$|[.,!?])', lambda m: create_wiki_link(m.group(1), 'secondary'), text)
    text = re.sub(r'\[3\]\[([^\]]+?)\](?=\s|$|[.,!?])', lambda m: create_wiki_link(m.group(1), 'tertiary'), text)
    
    # Clean up any remaining markers and whitespace
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return f'<div class="wiki-content">{text}</div>'

def validate_wiki_content(text, title):
    """Validate that the Wikipedia content is relevant to the query."""
    try:
        # Skip literary/fictional/media content
        irrelevant_terms = {
            'novel', 'fiction', 'literary', 'literature', 'story', 'stories', 
            'author', 'writer', 'poem', 'poetry', 'television', 'tv', 'series',
            'episode', 'show', 'movie', 'film', 'anime', 'manga', 'comic'
        }
        
        # Check title and first paragraph for irrelevant terms
        if any(term in title.lower() for term in irrelevant_terms):
            return False
        if any(term in text.lower()[:500] for term in irrelevant_terms):
            return False
            
        # Extract key terms from the content
        doc = nlp(text[:1000])  # Limit to first 1000 chars for performance
        content_entities = set([ent.text.lower() for ent in doc.ents])
        
        # Calculate relevance scores
        term_matches = sum(term.lower() in text.lower() for term in key_terms)
        entity_overlap = sum(1 for term in key_terms if any(term.lower() in entity for entity in content_entities))
        
        # Check for historical indicators
        historical_terms = {
            'history', 'century', 'ancient', 'period', 'era', 'historical',
            'empire', 'kingdom', 'dynasty', 'ruler', 'reign', 'conquest',
            'civilization', 'culture', 'society', 'development'
        }
        
        # Require stronger historical context
        historical_context_score = sum(1 for term in historical_terms if term in text.lower())
        
        # Article must have:
        # 1. Multiple term matches or entity overlaps
        # 2. Strong historical context
        # 3. No irrelevant terms
        is_relevant = (
            (term_matches >= 2 or entity_overlap >= 2) and
            historical_context_score >= 2
        )
        
        return is_relevant
        
    except Exception as e:
        return False  # Reject content if validation fails

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

def process_suggestions(response_text):
    """Extract and clean up suggestions from the response text."""
    suggestions = []
    lines = response_text.split('\n')
    
    for line in lines:
        if line.strip().startswith('[SUGGESTION]'):
            # Clean up the suggestion text
            suggestion = line.replace('[SUGGESTION]', '').strip()
            suggestion = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', suggestion)
            suggestion = re.sub(r'\s+', ' ', suggestion)
            if suggestion:
                suggestions.append(suggestion)
    
    return suggestions

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
            recent_messages = st.session_state.messages[-6:]  # Get last 6 messages
            conversation_context = "\nPrevious conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = re.sub(r'<[^>]+>', '', msg["content"])
                content = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\2', content)
                conversation_context += f"{role}: {content}\n"

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a historian specializing in providing accurate historical information. Your responses must:
1. Use ONLY information from the provided Wikipedia content
2. Stay focused on historical facts and developments
3. Consider the conversation history and previous questions when relevant
4. Mark important terms with:
   - [1][term] for major figures and events
   - [2][term] for dates and places
   - [3][term] for supporting details"""
                },
                {
                    "role": "user",
                    "content": f"""Use ONLY the Wikipedia content below to answer this history question. Consider the conversation history for context.

Wikipedia Content:
{wiki_content}

Previous Conversation:
{conversation_context}

Current Question: {prompt}

Mark important terms with [1], [2], or [3] markers and end with three follow-up questions on new lines starting with [SUGGESTION]"""
                }
            ],
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Split response and extract suggestions
        parts = response_text.split('\n')
        main_response = []
        suggestions = []
        
        # Process each line
        for part in parts:
            if part.strip().startswith('[SUGGESTION]'):
                suggestion = part.replace('[SUGGESTION]', '').strip()
                suggestion = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', suggestion)
                suggestion = re.sub(r'\s+', ' ', suggestion)
                if suggestion:
                    suggestions.append(suggestion)
            else:
                main_response.append(part)
        
        main_response = ' '.join(main_response).strip()
        
        # Store suggestions in session state
        st.session_state.suggestions = suggestions[:3]
        
        # If we don't have enough suggestions, generate some based on found articles
        if len(st.session_state.suggestions) < 3 and 'last_wiki_articles' in st.session_state:
            article_titles = list(st.session_state['last_wiki_articles'].keys())
            while len(st.session_state.suggestions) < 3 and article_titles:
                title = random.choice(article_titles)
                suggestion = f"Tell me more about {title}"
                if suggestion not in st.session_state.suggestions:
                    st.session_state.suggestions.append(suggestion)
                article_titles.remove(title)
        
        # If still not enough suggestions, add a generic one
        if len(st.session_state.suggestions) < 3:
            st.session_state.suggestions.append("What other historical events were significant during this period?")
        
        # Clean up formatting artifacts
        main_response = re.sub(r'PART \d:', '', main_response)
        main_response = re.sub(r'\*\*.*?\*\*', '', main_response)
        main_response = re.sub(r'\d\. ', '', main_response)
        main_response = re.sub(r'Follow-Up Questions:', '', main_response)
        
        # Process URLs
        main_response = re.sub(r'https?://\S+', '', main_response)
        main_response = re.sub(r'\(https?://[^)]+\)', '', main_response)
        
        # Clean up extra spaces and normalize whitespace
        main_response = re.sub(r'\s+', ' ', main_response)
        main_response = main_response.strip()
        
        # Let add_wiki_links handle the importance markers
        return add_wiki_links(main_response)
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def validate_content(prompt, response_text):
    """Validate that the AI response stays on topic and relevant to the prompt."""
    try:
        # Store original response with formatting
        original_response = response_text
        
        # Clean text only for validation purposes
        validation_text = re.sub(r'<[^>]+>', '', response_text)
        validation_text = re.sub(r'\[\d+\]\[([^\]]+)\]', r'\1', validation_text)
        
        # Extract key terms from the prompt
        prompt_doc = nlp(prompt)
        prompt_entities = set([ent.text.lower() for ent in prompt_doc.ents])
        prompt_nouns = set([token.text.lower() for token in prompt_doc if token.pos_ in ['PROPN', 'NOUN']])
        prompt_key_terms = prompt_entities.union(prompt_nouns)
        
        # Add historical context terms
        historical_terms = {'history', 'century', 'period', 'era', 'empire', 'kingdom', 'state', 'nation', 'population', 'people'}
        prompt_key_terms.update(historical_terms)
        
        # Extract key terms from the response
        response_doc = nlp(validation_text)
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

def post_process_response(text):
    """Ensure consistent formatting of important terms in the response."""
    # Extract entities and important terms
    doc = nlp(text)
    
    # Build a list of terms to mark with importance levels
    important_terms = []
    secondary_terms = []
    tertiary_terms = []
    
    # Identify important terms
    for ent in doc.ents:
        if ent.label_ in {'PERSON', 'EVENT', 'ORG', 'GPE', 'LOC'}:
            if len(ent.text) > 3:  # Avoid short terms
                if ent.label_ in {'PERSON', 'EVENT'}:
                    important_terms.append(ent.text)
                elif ent.label_ in {'ORG', 'GPE'}:
                    secondary_terms.append(ent.text)
                else:
                    tertiary_terms.append(ent.text)
    
    # Add dates to secondary terms
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            if any(c.isdigit() for c in ent.text):  # Only dates with numbers
                secondary_terms.append(ent.text)
    
    # Sort terms by length (longest first) to handle nested terms correctly
    important_terms.sort(key=len, reverse=True)
    secondary_terms.sort(key=len, reverse=True)
    tertiary_terms.sort(key=len, reverse=True)
    
    # Apply formatting
    for term in important_terms:
        text = re.sub(rf'\b{re.escape(term)}\b', f'[1][{term}]', text, flags=re.IGNORECASE)
    for term in secondary_terms:
        text = re.sub(rf'\b{re.escape(term)}\b', f'[2][{term}]', text, flags=re.IGNORECASE)
    for term in tertiary_terms:
        text = re.sub(rf'\b{re.escape(term)}\b', f'[3][{term}]', text, flags=re.IGNORECASE)
    
    return text

def get_ai_response(prompt, wiki_content):
    """Get response from selected AI model with follow-up suggestions."""
    try:
        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Get response
        if model_choice == "Deepseek (Requires API Key)":
            response = get_deepseek_response(prompt, wiki_content)
        else:
            response = get_groq_response(prompt, wiki_content)
        
        # Convert markers to HTML links
        html_response = add_wiki_links(response)
        
        # Add sources section
        if 'last_wiki_articles' in st.session_state:
            sources_html = '<div style="margin-top: 2rem; margin-bottom: 1rem;">'
            sources_html += '<details style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;">'
            sources_html += '<summary style="padding: 1rem; cursor: pointer; user-select: none; font-weight: 500; color: rgba(255, 255, 255, 0.8);">Content Used from Wikipedia</summary>'
            sources_html += '<div style="max-height: 300px; overflow-y: auto; padding: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">'
            
            for title, data in st.session_state['last_wiki_articles'].items():
                if data['summary'] or data['used_sections']:
                    sources_html += f'<div style="margin-bottom: 1.5rem;">'
                    sources_html += f'<div style="margin-bottom: 0.5rem;">'
                    sources_html += f'<a href="https://en.wikipedia.org/wiki/{title.replace(" ", "_")}" target="_blank" style="color: rgba(255, 255, 255, 0.8); text-decoration: none; border-bottom: 1px dotted rgba(255, 255, 255, 0.3); font-weight: 500;">{title}</a>'
                    sources_html += '</div>'
                    
                    if data['summary']:
                        sources_html += '<div style="margin-left: 1rem; margin-bottom: 0.5rem;">'
                        sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.3rem;"><em>From main article:</em></div>'
                        sources_html += f'<div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9em; line-height: 1.4;">{data["summary"][:200]}...</div>'
                        sources_html += '</div>'
                    
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
            
            # Add sources to the response
            if html_response.endswith('</div>'):
                html_response = html_response[:-6] + sources_html + '</div>'
            else:
                html_response += sources_html
                
        return html_response
            
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

def handle_chat():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            # Only show suggestions after the most recent assistant message
            if (message["role"] == "assistant" and 
                message == st.session_state.messages[-1] and 
                st.session_state.suggestions):
                
                # Create columns for suggestions
                cols = st.columns(3)
                for idx, suggestion in enumerate(st.session_state.suggestions):
                    if suggestion:  # Only create button if suggestion exists
                        if cols[idx].button(
                            suggestion,
                            key=f"suggestion_{idx}_{len(st.session_state.messages)}",
                            use_container_width=True,
                            type="secondary"
                        ):
                            # Handle suggestion click
                            st.session_state.messages.append({"role": "user", "content": suggestion})
                            st.rerun()

# Chat input and main chat loop
if prompt := st.chat_input("What would you like to know about history?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching Wikipedia and composing response..."):
            wiki_content = get_wikipedia_content(prompt)
            
            if wiki_content:
                response = get_ai_response(prompt, wiki_content)
            else:
                response = get_ai_response(prompt, "No direct Wikipedia article found for this query.")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Display chat history and handle suggestions
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        
        # Only show suggestions after the most recent assistant message
        if (message["role"] == "assistant" and 
            idx == len(st.session_state.messages) - 1 and 
            st.session_state.suggestions):
            
            # Create columns for suggestions
            cols = st.columns(3)
            for i, suggestion in enumerate(st.session_state.suggestions):
                if suggestion:  # Only create button if suggestion exists
                    if cols[i].button(
                        suggestion,
                        key=f"suggestion_{i}_{idx}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        # Add suggestion as user message
                        st.session_state.messages.append({"role": "user", "content": suggestion})
                        st.rerun()  # Rerun here to update chat before getting response 