import os
import requests
from dotenv import load_dotenv

load_dotenv()


def send_message(message: str, platform: str = "slack") -> dict:
    """
    Send a message to Slack or Microsoft Teams using a webhook.
    """

    if not message or not message.strip():
        return {
            "success": False,
            "platform": platform,
            "error": "Message cannot be empty."
        }

    platform = platform.lower().strip()

    if platform == "slack":
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if not webhook_url:
            return {
                "success": False,
                "platform": "slack",
                "error": "SLACK_WEBHOOK_URL is not configured."
            }

        payload = {
            "text": message
        }

    elif platform == "teams":
        webhook_url = os.getenv("TEAMS_WEBHOOK_URL")

        if not webhook_url:
            return {
                "success": False,
                "platform": "teams",
                "error": "TEAMS_WEBHOOK_URL is not configured."
            }

        payload = {
            "text": message
        }

    else:
        return {
            "success": False,
            "platform": platform,
            "error": "Unsupported platform. Use 'slack' or 'teams'."
        }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        return {
            "success": True,
            "platform": platform,
            "status_code": response.status_code,
            "message": "Message sent successfully."
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "platform": platform,
            "error": str(error)
        }