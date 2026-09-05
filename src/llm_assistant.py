import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

class LLMAssistant:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            self.client = Groq(api_key=api_key)
            self.model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

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
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                return response.choices[0].message.content
                
            elif self.provider == "gemini":
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=300,
                    )
                )
                return response.text
                
        except Exception as e:
            return f"Error generating response via {self.provider.upper()}: {str(e)}"