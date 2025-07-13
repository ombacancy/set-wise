# Set-Wise

Set-Wise is an AI-based gym assistant application designed to help users track workouts, manage fitness goals, and receive personalized training recommendations through natural conversation.

## Features

- **Personalized Fitness Coaching**: Conversational AI coach that adapts to user goals and needs
- **User Profile Management**: Tracks height, weight, age, and fitness level metrics
- **Workout Tracking**: Records and analyzes completed workouts for future reference
- **Goal Setting and Monitoring**: Helps establish and track fitness objectives
- **Health Issue Awareness**: Adapts recommendations based on injuries or limitations
- **Workout Analytics**: Provides summary reports of training frequency and patterns
- **Personalized Recommendations**: Suggests workouts based on goals, history, and physical attributes

## Technical Architecture

Set-Wise leverages several advanced technologies:

- **Vector Databases**: Stores and retrieves information about health issues, fitness goals, and workout history
- **LLM Integration**: Uses Groq for natural language processing and generation
- **LangChain & LangGraph**: Manages conversation flow and tool execution
- **FastAPI**: Backend REST API for client-server communication

## API Endpoints

- `/api/users/register`: Create new user profiles
- `/api/users/login`: Authenticate existing users
- `/api/users/{user_id}/profile`: Retrieve user information
- `/api/chat/{user_id}`: Process messages with the AI coach
- `/api/chat/{user_id}/history`: View conversation history
- `/api/chat/{user_id}/clear`: Clear conversation history
- `/api/quick-prompts`: Get suggested prompts for the chat

## Data Management

The system stores four primary data types:

1. **Health Records**: Information about injuries and physical limitations
2. **Fitness Goals**: Short and long-term objectives
3. **Workout History**: Previously recommended workouts
4. **Workout Logs**: User-reported completed exercises

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
