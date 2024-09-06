from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
import pyrebase
from urllib.parse import urldefrag
import requests
import re
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
    "apiKey": os.getenv('FIREBASE_API'),
    "authDomain": os.getenv('FIREBASE_AUTH'),
    "databaseURL": os.getenv('DB_URL'),
    "projectId": os.getenv('FIREBASE_ID'),
    "storageBucket": os.getenv('STORAGE_BUCKET'),
    "messagingSenderId": os.getenv('MESSAGING_ID'),
    "appId": os.getenv('APP_ID'),
    "measurementId": os.getenv('MEASUREMENT_ID')
}
llm1 = ChatGoogleGenerativeAI(api_key = geminiAPI, model = 'gemini-1.5-flash')
llm2 = ChatGoogleGenerativeAI(api_key = geminiAPI, model = 'gemini-pro', convert_system_message_to_human = True)
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()
class ImageRequest(BaseModel):
    id : str
    query: str
    images: List[str]

def check_query(query):
  output = llm2.invoke(
    [
        SystemMessage(content = f"""You are a system that determines if a given query is referring to an uploaded image and any other image or if it is a standalone query. Your task is to analyze the query and respond with either "yes" or "no" based on the following conditions:

                                    Yes: If the query is referring to or asking about the uploaded image or any other image.
                                    No: If the query is a standalone, unrelated question.
                                    Respond with only "yes" or "no"."""),
        HumanMessage(content = f"""{query}""")
    ])
  if output.content == "yes":
    return True
  else:
    return False
def extract_title_and_questions(input_string):
    title_match = re.search(r"Title\s*:\s*(.*)", input_string)
    title = title_match.group(1).strip() if title_match else None
    questions = re.findall(r"\d+\.\s*(.*)", input_string)

    return title, questions
def generate_answer(query):
    output = llm2.invoke([
            SystemMessage(content = f"""Give response in conversational way.Only If anyone asks something related to you use the context "Am a Image Recognizing Conversational ChatBot my name is DrowserPandi" to answer or else just answer the query"""),
            HumanMessage(content = f"""{query}""")
        ])
    return output
def generate_questions(response):
    output = llm2.invoke(
    [
        SystemMessage(content = f"""Given a query generate a title and a list of questions related to the query. The expected output format is:
                                    Title : <generated title>
                                    Questions : [<generated questions1>,<generated questions2>,<generated questions3>...]"""),
        HumanMessage(content = f"""{response}""")
    ])
    title, questions = extract_title_and_questions(output.content)
    return title,questions
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
def put_index(uid,index):
  db.child("Users").child(uid).child("index").set(index)

def fetch_context(uid):
  context = db.child("Users").child(uid).child("context").get().val()
  return context
def fetch_index(uid):
    index = db.child("Users").child(uid).child("index").get().val()
    if index == None:
        index = 0
    else:
        index = int(index)
    return index

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
    flag = check_query(query)
    index = 0
    if flag:
        if '@' in query:
            index = query_parser(query)
        else:
            index = fetch_index(uid)
        context = fetch_context(uid)
        prompt = generate_prompt(query, context)
        response = requests.get(request.images[index])
        image = Image.open(BytesIO(response.content))
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        path = "download123.jpg"
        image.save(path)
        message = HumanMessage(
            content = [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': path}
            ]
        )
        response = llm1.invoke([message])
    else:
        response = generate_answer(query)
    title,questions=generate_questions(response.content)
    put_context(uid,query,response.content)
    put_index(uid,index)
    return {"title":title,"questions":questions,"response":response.content}
    
