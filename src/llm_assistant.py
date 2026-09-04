import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

class LLMAssistant:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key":
            print("WARNING: GROQ_API_KEY is not set correctly in .env!")
            
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def assist(self, ticket, queue, priority, sla):
        prompt = f"""You are a customer support assistant.

Ticket: {ticket}
Predicted Queue: {queue}
Predicted Priority: {priority}
SLA: {sla}

Provide:
SUMMARY:
A concise summary of the issue.

RESPONSE:
A professional response draft for a support agent.
Do not invent facts.
Do not claim an action was completed."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"