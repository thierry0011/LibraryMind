
1

Automatic Zoom
Module 10 Lab — LibraryMind: Lab write up
Page 1






M O D U L E   L a b
Module 10 Lab: LibraryMind
Lab Assignment Brief — AI-Powered Intelligent Library Assistant

Python Backend and AI Application

Module 10 Lab — LibraryMind: Lab write up
Page 2
Lab Overview
This  lab  consolidates  everything  you  have  learnt  across  module  10  content  three  into  a  single,
production-grade  application.  You  will  design,  implement,  and  deliver  a  working  AI-powered
backend service from scratch.
The Project: “LibraryMind”  An Intelligent Library Assistant
You will build LibraryMind, an AI-powered backend service for a public library. The system allows
patrons to:
• Search  the  library  catalogue  using  natural  language  questions  instead  of  exact  keyword
matches
• Get intelligent book recommendations based on their stated interests and preferences
• Ask detailed questions about books in the collection and receive grounded answers with
source citations
• Summarise collections of book reviews and extract key themes automatically
• Chat with an AI librarian that remembers the conversation context across multiple turns
• Submit support tickets that are automatically classified by category, priority, and sentiment
LibraryMind is the ideal capstone because it exercises every core skill from the training programme:
multi-provider  AI  integration,  vector-based  semantic  search,  retrieval-augmented  generation,
prompt engineering for structured outputs, conversational memory management, and production
concerns like caching, rate limiting, and cost tracking.
What You Will Build
Component  Purpose
Multi-Provider AI Layer  Abstraction over OpenAI, and Claude automatic fallback when one fails
Knowledge Base  Vector database storing book descriptions, searchable by meaning
RAG Engine  Retrieval-augmented question answering grounded in the catalogue
AI Librarian Chatbot  Multi-turn conversational agent with persistent memory
Classification Service  Auto-categorise and route member support tickets
Summarisation Service  Condense book reviews into themes, sentiment, and recommendations
REST API  FastAPI  application  exposing  all  functionality  through  documented
endpoints
Caching & Rate Limiting  Response  caching  and  request  throttling  to  protect  budget  and
performance
Usage Tracker  Token counting and cost estimation for every AI call
