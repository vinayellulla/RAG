from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # loads GROQ_API_KEY from .env
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
print(llm.invoke("In one sentence, what does an HTTP  400 error mean?").content)


