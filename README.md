Health-sql-bot
Disease Treatment Query Assistant

This application uses Streamlit, SQLite, and LLaMA 2 (via LangChain's OllamaLLM) to convert English questions into SQL queries and fetch relevant data from a health database.

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Health-sql-bot-2.O Disease Assistant with Gemini AI
Advanced Disease Treatment and Health Query Assistant

This enhanced Streamlit app leverages Google Gemini generative AI alongside SQLite to provide detailed responses to health-related questions.

Features
Connects to a local SQLite database containing disease and treatment information.

Uses Google Gemini AI for natural language understanding and generating responses.

Employs fuzzy matching to identify diseases from user queries, even with typos.

Maintains persistent chat history stored in SQLite per user.

Provides tailored treatment advice, symptom explanations, and general health info.

Handles follow-up questions with context awareness.

Offers subscription information on request.

Clean and interactive UI with Streamlit.

Technologies
Streamlit — Web app framework for Python

SQLite — Lightweight database for treatment data and chat history

Google Gemini AI — Generative AI model accessed via google.generativeai Python SDK

LangChain — For chat message history management

RapidFuzz — For fuzzy string matching to identify diseases

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Health-sql-bot-3.O
Chat with your medical PDFs using Gemini 1.5 Flash
This version of the Health-sql-bot project allows users to upload medical documents (PDFs) and ask questions directly about their contents. It uses Google's Gemini 1.5 Flash model to understand queries and provide detailed, context-aware answers based on the PDF text.

