# ai/api_connector.py

import requests
from typing import Dict, Any, List, Optional


class APIConnector:
    """Connector class to communicate with the FastAPI backend"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new user"""
        url = f"{self.base_url}/api/users/register"
        try:
            response = requests.post(url, json=user_data)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_message = error_data.get("detail",
                                                   f"Registration failed with status code: {e.response.status_code}")
                except ValueError:
                    error_message = f"Registration failed with status code: {e.response.status_code}"
            else:
                error_message = "Cannot connect to API server"

            print(f"API Error: {error_message}")
            raise Exception(f"Error: {error_message}")

    def login_user(self, user_id: str) -> Dict[str, Any]:
        """Login an existing user"""
        url = f"{self.base_url}/api/users/login"
        response = requests.post(url, json={"user_id": user_id})

        if response.status_code == 200:
            return response.json()
        else:
            error_message = response.json().get("detail", "Login failed")
            raise Exception(f"Error: {error_message}")

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        url = f"{self.base_url}/api/users/{user_id}/profile"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            error_message = response.json().get("detail", "Failed to get profile")
            raise Exception(f"Error: {error_message}")

    def chat_with_coach(self, user_id: str, message: str) -> Dict[str, Any]:
        """Send a message to the AI fitness coach"""
        url = f"{self.base_url}/api/chat/{user_id}"
        response = requests.post(url, json={"content": message})

        if response.status_code == 200:
            return response.json()
        else:
            error_message = response.json().get("detail", "Chat failed")
            raise Exception(f"Error: {error_message}")

    def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a user"""
        url = f"{self.base_url}/api/chat/{user_id}/history"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()["messages"]
        else:
            error_message = response.json().get("detail", "Failed to get chat history")
            raise Exception(f"Error: {error_message}")

    def clear_chat_history(self, user_id: str) -> Dict[str, Any]:
        """Clear chat history for a user"""
        url = f"{self.base_url}/api/chat/{user_id}/clear"
        response = requests.delete(url)

        if response.status_code == 200:
            return response.json()
        else:
            error_message = response.json().get("detail", "Failed to clear chat history")
            raise Exception(f"Error: {error_message}")

    def get_quick_prompts(self) -> List[Dict[str, Any]]:
        """Get quick prompts for the chat"""
        url = f"{self.base_url}/api/quick-prompts"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            return [
                {"title": "Set fitness goals", "content": "I want to focus on fat loss and muscle toning."},
                {"title": "Report injury", "content": "My right shoulder has been hurting during overhead exercises."},
                {"title": "Request workout", "content": "Can you suggest a home workout with minimal equipment?"},
                {"title": "Ask about nutrition", "content": "What should I eat to support muscle growth?"}
            ]

    def check_api_status(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}: {str(e)}")
