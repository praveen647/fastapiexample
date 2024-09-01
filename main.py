from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI

app = FastAPI()

import getpass
import os
geminiAPI = os.getenv('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(api_key = geminiAPI, model = 'gemini-pro', temperature = 0.9)
@app.get("/")
async def root():
    return {"message": "Hello World from Mugundhan"}

@app.get("/query")
def read_item(q:str):
    response = llm.invoke(q)
    return response
