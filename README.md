# Historical Chat Bot

An AI-powered chat interface that combines Wikipedia knowledge with advanced language models to provide comprehensive answers to historical questions.

Go to historical-chat.streamlit.app to use the free webhosted app.

## Features

### Core Functionality
- Historical Q&A powered by Wikipedia integration and AI models
- Choice between Groq (free) and Deepseek (requires API key) models
- British English audio playback of responses using Google Text-to-Speech
- Smart follow-up questions for deeper exploration
- Clean, responsive dark-themed interface

### User Experience
- Secure user authentication with login/signup
- Conversation management (save, load, delete)
- API key storage for returning users
- Interactive follow-up question buttons
- Audio playback button for responses
- Response reshuffling for alternative perspectives

## Technical Stack

- **Frontend**: Streamlit
- **Database**: SQLite with SQLAlchemy ORM
- **APIs**: 
  - Wikipedia API for content retrieval
  - Groq/Deepseek for AI responses
  - Google Text-to-Speech for audio
- **Security**: bcrypt for password hashing
- **Dependencies**: Python 3.7+, see requirements.txt

## Local Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
5. Add your API keys to the `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here  # Optional
   ```

## Running the Application

Start the application in development mode:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Sign up for an account or log in
2. Choose your preferred AI model (Groq or Deepseek)
3. Enter your API key(s) in the sidebar
4. Start asking historical questions!
5. Use the audio button to listen to responses
6. Click follow-up questions or use the shuffle button for different perspectives
7. Your conversations are automatically saved

## Development Notes

### Key Learnings
- Streamlit's session state management is crucial for multi-user support
- SQLAlchemy provides robust database interactions
- Proper error handling for API calls improves reliability
- Clean separation of concerns between UI, database, and API logic

### Known Considerations
- Audio playback uses gTTS (no API key required)
- Database is SQLite for simplicity, may need migration for production
- API keys are stored securely but transmitted in plaintext to APIs
- Development mode uses local database

### Future Improvements
- Add conversation search/filtering
- Implement conversation export (PDF/text)
- Add user preferences (theme, language)
- Enhance error handling and user feedback
- Add conversation sharing between users

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with assistance from Claude AI
- Uses Groq and Deepseek for AI capabilities
- Wikipedia for historical content
- Streamlit for the amazing web framework 
