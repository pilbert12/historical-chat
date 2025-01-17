# Historical Chat Bot

A chatbot that combines Wikipedia knowledge with Deepseek AI to provide comprehensive answers to historical questions.

## Features

- Search Wikipedia for historical information
- Enhanced responses using Deepseek AI
- Clean and intuitive Streamlit interface
- Real-time chat history
- Combines factual information with AI insights

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Add your Deepseek API key to the `.env` file:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

## Running the Application

Run the application using Streamlit:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Type your historical question in the chat input
2. The bot will:
   - Search Wikipedia for relevant information
   - Process the information using Deepseek AI
   - Provide a comprehensive response
3. Chat history is maintained during your session

## Note

Make sure you have a valid Deepseek API key. You can obtain one from the Deepseek platform. 