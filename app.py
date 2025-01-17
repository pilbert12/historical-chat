import streamlit as st
import wikipedia
import requests
import json
import time
import re
import spacy
import concurrent.futures
from groq import Groq

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.warning("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Configure Wikipedia
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)

# Initialize session state if needed
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

# Set up the page
st.set_page_config(page_title="Historical Chat Bot", page_icon="📚", layout="wide")
st.title("Historical Chat Bot")
st.write("Ask me anything about history! I'll combine Wikipedia knowledge with AI insights.")

def check_api_key(model_choice):
    """Check if the appropriate API key is available for the selected model."""
    if model_choice == "Deepseek (Requires API Key)":
        api_key = st.session_state.get('DEEPSEEK_API_KEY')
        if not api_key:
            st.error("Please enter your Deepseek API key in the sidebar to use AI-powered search.")
            return False
    else:  # Groq
        api_key = st.session_state.get('GROQ_API_KEY')
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar to use AI-powered search.")
            return False
    return True

def get_wikipedia_content(query):
    """Search Wikipedia and get content for the query using AI-driven search."""
    try:
        # Create a placeholder for showing search progress
        search_progress = st.empty()
        search_progress.markdown("🤔 Analyzing query to develop search strategy...")
        
        # Get model choice and check API key
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        if not check_api_key(model_choice):
            return None
        
        # Try to get AI-generated search terms
        try:
            search_progress.markdown("🧠 Generating intelligent search strategy...")
            search_terms = get_ai_search_terms(query, model_choice)
            
            if not search_terms:
                search_progress.warning("AI search term generation failed, using fallback strategy...")
                search_terms = get_default_search_terms(query)
            else:
                search_progress.success(f"Generated {len(search_terms)} search terms")
                
        except Exception as e:
            search_progress.warning(f"Error generating search terms: {str(e)}\nUsing fallback strategy...")
            search_terms = get_default_search_terms(query)
        
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
            # Check overall search timeout (10 minutes)
            if time.time() - search_start_time > 600:
                search_progress.warning("⚠️ Search timeout reached. Using available results...")
                break
                
            # Update progress less frequently to avoid UI lag
            current_time = time.time()
            if current_time - last_progress_update >= 0.5:
                search_progress.markdown(
                    f"🔍 Searching... (Term {term_index + 1} of {len(search_terms)})\n\n"
                    f"Current term: '{term}'\n\n"
                    f"Articles found: {relevant_articles_found} relevant / {processed_count} processed"
                )
                last_progress_update = current_time
            
            try:
                # Get initial search results using requests directly
                try:
                    search_url = f"https://en.wikipedia.org/w/api.php"
                    search_params = {
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": term,
                        "srlimit": 20
                    }
                    response = requests.get(search_url, params=search_params)
                    search_data = response.json()
                    search_results = [result['title'] for result in search_data['query']['search']]
                except Exception as wiki_error:
                    st.error(f"Wikipedia search error: {str(wiki_error)}")
                    continue
                
                term_start_time = time.time()
                
                # Process each potential article
                for title in search_results:
                    # Check per-term timeout (90 seconds)
                    if time.time() - term_start_time > 90:
                        break
                        
                    processed_count += 1
                    
                    if title in found_articles:
                        continue
                        
                    try:
                        # Try to get the page content using requests directly
                        try:
                            page_url = f"https://en.wikipedia.org/w/api.php"
                            page_params = {
                                "action": "query",
                                "format": "json",
                                "prop": "extracts|info",
                                "exintro": True,
                                "explaintext": True,
                                "inprop": "url",
                                "titles": title
                            }
                            page_response = requests.get(page_url, params=page_params)
                            page_data = page_response.json()
                            page_id = list(page_data['query']['pages'].keys())[0]
                            page = page_data['query']['pages'][page_id]
                            
                            if 'missing' in page:
                                continue
                                
                            summary = page.get('extract', '')
                            url = page.get('fullurl', '')
                            
                        except Exception as e:
                            continue
                            
                        # Get full content for validation
                        content_to_validate = f"Title: {title}\n\nSummary: {summary}\n\nURL: {url}"
                        
                        # Full AI validation
                        try:
                            with st.spinner(f"Validating: {title}"):
                                is_relevant = validate_article_with_timeout({"title": title, "summary": summary, "url": url}, query, model_choice)
                        except Exception as e:
                            continue
                            
                        if is_relevant:
                            relevant_articles_found += 1
                            
                            # Store article info
                            found_articles[title] = {
                                'summary': summary[:1000],
                                'used_sections': [],
                                'relevance': 0.8,
                                'url': url
                            }
                            
                            # Add summary to content
                            if summary not in seen_content:
                                seen_content.add(summary)
                                wiki_content.append(f"From article '{title}':\n{summary[:1000]}")
                            
                            # If we have enough high-quality articles, we can stop
                            if relevant_articles_found >= 10:
                                break
                                
                    except Exception as e:
                        continue
                        
                # If we have enough articles, we can stop
                if relevant_articles_found >= 10:
                    break
                    
            except Exception as e:
                continue
            
            # If we've processed too many articles without finding any relevant ones,
            # continue to the next search term
            if processed_count >= 100 and relevant_articles_found == 0:
                continue
        
        # Show final stats
        search_progress.markdown(
            f"✅ Search complete!\n\n"
            f"• Terms tried: {term_index + 1} of {len(search_terms)}\n"
            f"• Articles found: {len(found_articles)} relevant\n"
            f"• Total processed: {processed_count}\n"
            f"• Search time: {time.time() - search_start_time:.1f}s"
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

# Rest of your code... 