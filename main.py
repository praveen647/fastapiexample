from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
import pyrebase
from urllib.parse import urldefrag
import requests
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List

app = FastAPI()

import getpass
import os
geminiAPI = os.getenv('GEMINI_API_KEY')
config = {
    "apiKey": "AIzaSyAXyeap4l3_gNoTFR4-YX3MJH8PE-9Qn1w",
    "authDomain": "example-fastapi-f5a95.firebaseapp.com",
    "databaseURL": "https://example-fastapi-f5a95-default-rtdb.firebaseio.com/",
    "projectId": "example-fastapi-f5a95",
    "storageBucket": "example-fastapi-f5a95.appspot.com",
    "messagingSenderId": "1024331035169",
    "appId": "1:1024331035169:web:42773da2833da4e5a92960",
    "measurementId": "G-5MV233N64X"
}
llm = ChatGoogleGenerativeAI(api_key = geminiAPI, model = 'gemini-1.5-flash')
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()
class ImageRequest(BaseModel):
    id : str
    query: str
    images: List[str]
def put_context(uid,query,response):
  context = fetch_context(uid)
  if context is None:
        context = []
  context.append({"query": query, "response": response})
  db.child("Users").child(uid).child("context").set(context)

def generate_prompt(query, context):
  if context == None:
    prompt = f"""
    Answer the following query:
    {query}
    """
  else:
    context_str = " ".join([f"Query: {item['query']} Response: {item['response']} \n" for item in context])
    prompt = f"""
    Given the information of the all the previous conversation below:
    {context_str}
    _________________________________________________________________
    Answer the following query:
    {query}
    """
  return prompt

def fetch_context(uid):
  context = db.child("Users").child(uid).child("context").get().val()
  return context


def query_parser(query):
  if '@' in query:
    index = int(query.split('@')[1][-1])
  return index


@app.get("/")
async def root():
    return {"message": "Hello World from Mugundhan"}

@app.post("/generate")
def generate_response(request:ImageRequest):
    uid = request.id
    query = request.query
    if '@' in query:
        index = query_parser(query)
    else:
        index = -1
    context = fetch_context(uid)
    prompt = generate_prompt(query, context)
    response = requests.get(request.images[index])
    image = Image.open(BytesIO(response.content))
    path = "download123.jpg"
    image.save(path)
    message = HumanMessage(
        content = [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': path}
        ]
    )
    response = llm.invoke([message])
    put_context(uid,query,response.content)
    return {"response":response.content}
    
