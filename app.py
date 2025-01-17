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

def validate_article_with_timeout(page, query, model_choice, timeout=30):
    """Validate article relevance with a timeout."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(validate_wiki_content, page.content[:2000], page.title, get_key_terms(query))
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return False
    except Exception as e:
        return False

def validate_wiki_content(text, title, key_terms=None):
    """Validate that the Wikipedia content is relevant to the query using AI."""
    try:
        # Build validation prompt
        validation_prompt = f"""Evaluate if this Wikipedia content would help answer questions about: {', '.join(key_terms) if key_terms else 'None provided'}

Article Title: {title}

Content:
{text[:2000]}

Your task is to determine if this content contains valuable historical information relevant to the search terms.
Consider both direct relevance and important contextual/background information that would enrich understanding.

Respond with a JSON object containing your evaluation:
{{
    "is_relevant": true/false,
    "relevance_score": 0-100,
    "reasoning": "Brief explanation of your decision"
}}"""

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
                            'content': 'You are an expert at evaluating historical content relevance.'
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
                        "content": "You are an expert at evaluating historical content relevance."
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
        
        # Accept content that is either highly relevant or provides important context
        return result.get('is_relevant', False) and result.get('relevance_score', 0) >= 40
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return False

def get_ai_search_terms(query, model_choice):
    """Generate intelligent search terms using AI."""
    try:
        # Build search strategy prompt
        strategy_prompt = f"""As a historical research assistant, analyze this query and develop a comprehensive search strategy.

Query: {query}

Create a thorough search strategy that will help find relevant Wikipedia articles. Generate AT LEAST 15 search terms considering:
1. Key historical figures, events, and concepts
2. Relevant time periods and locations
3. Related historical contexts and themes
4. Alternative names or terms that might be used
5. Broader historical context and related topics
6. Contemporary events and influences
7. Economic, social, and political aspects
8. Regional and global perspectives

Your response MUST include at least 15 search terms, from most specific to most general.

Respond with ONLY a JSON array of search terms. Example:
["term 1", "term 2", "term 3", ...]"""

        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        # Get AI-generated search strategy
        if model_choice == "Deepseek (Requires API Key)":
            api_key = st.session_state.get('DEEPSEEK_API_KEY')
            if not api_key:
                return None
                
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
                            'content': 'You are a historical research assistant. Generate at least 15 search terms.'
                        },
                        {
                            'role': 'user',
                            'content': strategy_prompt
                        }
                    ],
                    'temperature': 0.7
                }
            )
            search_terms = json.loads(response.json()['choices'][0]['message']['content'])
            
        else:
            api_key = st.session_state.get('GROQ_API_KEY')
            if not api_key:
                return None
                
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a historical research assistant. Generate at least 15 search terms."
                    },
                    {
                        "role": "user",
                        "content": strategy_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                top_p=1
            )
            
            search_terms = json.loads(completion.choices[0].message.content)
        
        # Validate and clean up search terms
        if not isinstance(search_terms, list):
            raise ValueError("AI did not return a list of search terms")
            
        # Remove any empty or non-string terms
        search_terms = [str(term).strip() for term in search_terms if term and isinstance(term, (str, int, float))]
        
        # Remove duplicates while preserving order
        seen = set()
        search_terms = [x for x in search_terms if not (x.lower() in seen or seen.add(x.lower()))]
        
        # Add default terms if we don't have enough
        if len(search_terms) < 15:
            default_terms = get_default_search_terms(query)
            search_terms.extend([t for t in default_terms if t not in search_terms])
        
        # Ensure we have at least 15 terms
        if len(search_terms) < 15:
            raise ValueError("Not enough search terms generated")
            
        return search_terms
        
    except Exception as e:
        return None

def get_ai_response(prompt, wiki_content):
    """Get response from selected AI model with follow-up suggestions."""
    try:
        # Build the prompt
        full_prompt = f"""You are a knowledgeable historical expert. Using the Wikipedia content below, provide an insightful response about: {prompt}

Wikipedia Content:
{wiki_content}

Your goal is to craft an engaging, informative response that:
- Draws from the provided Wikipedia content
- Highlights key historical figures, events, and concepts
- Provides relevant dates and contextual information
- Makes connections between different aspects of the topic
- Suggests natural follow-up questions to explore further

Feel free to structure and style your response in the most effective way. Mark important terms with [1][term], [2][term], or [3][term] based on their significance.

End your response with three follow-up questions, each on a new line starting with [SUGGESTION]."""

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
        
        # If content is not valid, get a new response with correction prompt
        if not is_valid and correction_prompt:
            if model_choice == "Deepseek (Requires API Key)":
                response = get_deepseek_response(correction_prompt, wiki_content)
            else:
                response = get_groq_response(correction_prompt, wiki_content)
        
        # Add sources if available
        if 'last_wiki_articles' in st.session_state and st.session_state['last_wiki_articles']:
            sources_html = '<div style="margin-top: 2rem; margin-bottom: 1rem;">'
            sources_html += '<details style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px;">'
            sources_html += '<summary style="padding: 1rem; cursor: pointer; user-select: none; font-weight: 500; color: rgba(255, 255, 255, 0.8);">Content Used from Wikipedia</summary>'
            sources_html += '<div style="max-height: 300px; overflow-y: auto; padding: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">'
            
            for title, data in st.session_state['last_wiki_articles'].items():
                sources_html += f'<div style="margin-bottom: 1.5rem;">'
                sources_html += f'<div style="margin-bottom: 0.5rem;">'
                sources_html += f'<a href="https://en.wikipedia.org/wiki/{title.replace(" ", "_")}" target="_blank" style="color: rgba(255, 255, 255, 0.8); text-decoration: none; border-bottom: 1px dotted rgba(255, 255, 255, 0.3); font-weight: 500;">{title}</a>'
                sources_html += '</div>'
                
                if data.get('summary'):
                    sources_html += '<div style="margin-left: 1rem; margin-bottom: 0.5rem;">'
                    sources_html += f'<div style="color: rgba(255, 255, 255, 0.6); font-size: 0.9em; margin-bottom: 0.3rem;"><em>From main article:</em></div>'
                    sources_html += f'<div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9em; line-height: 1.4;">{data["summary"][:200]}...</div>'
                    sources_html += '</div>'
                
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
            
            if '</div>' in response:
                response = response.replace('</div>', sources_html + '</div>', 1)
            else:
                response = f'<div>{response}{sources_html}</div>'
        
        return response
            
    except Exception as e:
        return f"Error communicating with AI model: {str(e)}"

def get_key_terms(text):
    """Extract key terms from text using spaCy."""
    doc = nlp(text)
    key_terms = []
    
    # Add named entities
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'EVENT', 'FAC', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON', 'PRODUCT']:
            key_terms.append(ent.text)
    
    # Add noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Limit to phrases of 3 words or less
            key_terms.append(chunk.text)
    
    # Remove duplicates while preserving order
    seen = set()
    key_terms = [x for x in key_terms if not (x.lower() in seen or seen.add(x.lower()))]
    
    return key_terms

def get_deepseek_response(prompt, wiki_content):
    """Get response from Deepseek API."""
    try:
        api_key = st.session_state.get('DEEPSEEK_API_KEY')
        if not api_key:
            return "Please enter your Deepseek API key in the sidebar."
            
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
                        'content': 'You are a knowledgeable historical expert.'
                    },
                    {
                        'role': 'user',
                        'content': f"Using ONLY the Wikipedia content below, provide an insightful response about: {prompt}\n\nWikipedia Content:\n{wiki_content}"
                    }
                ],
                'temperature': 0.7
            }
        )
        
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        return f"Error communicating with Deepseek API: {str(e)}"

def get_groq_response(prompt, wiki_content):
    """Get response from Groq API."""
    try:
        api_key = st.session_state.get('GROQ_API_KEY')
        if not api_key:
            return "Please enter your Groq API key in the sidebar."
            
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable historical expert."
                },
                {
                    "role": "user",
                    "content": f"Using ONLY the Wikipedia content below, provide an insightful response about: {prompt}\n\nWikipedia Content:\n{wiki_content}"
                }
            ],
            temperature=0.7,
            max_tokens=4096,
            top_p=1
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def validate_content(prompt, response):
    """Validate that the response is relevant and accurate."""
    try:
        # Get model choice from session state
        model_choice = st.session_state.get('model_choice', "Groq (Free)")
        
        validation_prompt = f"""Analyze this response to the historical question: "{prompt}"

Response to validate:
{response}

Your task:
1. Check if the response stays focused on the question
2. Verify that no unrelated topics are introduced
3. Ensure the response provides relevant historical information

If the response is valid, respond with:
{{"is_valid": true}}

If the response needs correction, respond with:
{{"is_valid": false, "correction_prompt": "Your suggested correction prompt"}}"""
        
        if model_choice == "Deepseek (Requires API Key)":
            api_key = st.session_state.get('DEEPSEEK_API_KEY')
            if not api_key:
                return True, None
                
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
                            'content': 'You are an expert at validating historical content.'
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
                return True, None
                
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at validating historical content."
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
        
        return result.get('is_valid', True), result.get('correction_prompt', None)
        
    except Exception as e:
        st.error(f"Error validating content: {str(e)}")
        return True, None

# Rest of your code... 