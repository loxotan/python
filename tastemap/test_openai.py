import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5.4-mini",
    input="한국어로 짧게 답해줘. API 연결 테스트 성공?"
)

print(response.output_text)