import os
from openai import OpenAI
from src.explanation.templates import ExplanationTemplates
from src.config import OPENAI_API_KEY, DEFAULT_MODEL

# -----------------------------
# Initialize OpenAI client
# -----------------------------

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

print("API key detected:", OPENAI_API_KEY[:8] + "********")

client = OpenAI(
    api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a system test."},
            {"role": "user", "content": "Reply with the word SUCCESS only."}
        ],
        max_tokens=10,
        temperature=0.0,
    )

    print("✅ API call succeeded")
    print("Response:", response.choices[0].message.content.strip())

except Exception as e:
    print("❌ API call failed")
    print("Error type:", type(e).__name__)
    print("Error message:", str(e))
