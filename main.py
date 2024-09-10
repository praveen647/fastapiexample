from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI, HTTPException
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

llm1 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-1.5-flash')
llm2 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-pro', convert_system_message_to_human=True)
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

class ImageRequest(BaseModel):
    id: str
    query: str
    images: List[str]

def check_query(query):
    try:
        output = llm2.invoke([
            SystemMessage(content="""You are a system that determines if a given query is referring to an uploaded image or if it is a standalone query. 
                                    Respond with "yes" if it refers to an image, "no" otherwise."""),
            HumanMessage(content=query)
        ])
        return output.content == "yes"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking query: {str(e)}")

def extract_title_and_questions(input_string):
    try:
        title_match = re.search(r"Title\s*:\s*(.*)", input_string)
        title = title_match.group(1).strip() if title_match else None
        questions = re.findall(r"\d+\.\s*(.*)", input_string)
        return title, questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting title and questions: {str(e)}")

def generate_answer(query):
    try:
        output = llm1.invoke([
            SystemMessage(content="""You are a conversational chatbot. Answer the user's question simply and relevantly."""),
            HumanMessage(content=query)
        ])
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def generate_questions(response):
    try:
        output = llm2.invoke([
            SystemMessage(content="""Generate a title and a list of questions based on the query. 
                                     Format: Title: <title>, Questions: [<question1>, <question2>, ...]"""),
            HumanMessage(content=response)
        ])
        title, questions = extract_title_and_questions(output.content)
        return title, questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

def put_context(uid, query, response):
    try:
        context = fetch_context(uid)
        if context is None:
            context = []
        context.append({"query": query, "response": response})
        db.child("Users").child(uid).child("context").set(context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving context: {str(e)}")

def generate_prompt(query, context):
    try:
        if context is None:
            prompt = f"Answer the following query:\n{query}"
        else:
            context_str = " ".join([f"Query: {item['query']} Response: {item['response']}\n" for item in context])
            prompt = f"Given the previous conversation:\n{context_str}\nAnswer the following query:\n{query}"
        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")

def put_index(uid, index):
    try:
        db.child("Users").child(uid).child("index").set(index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving index: {str(e)}")

def fetch_context(uid):
    try:
        return db.child("Users").child(uid).child("context").get().val()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching context: {str(e)}")

def fetch_index(uid):
    try:
        index = db.child("Users").child(uid).child("index").get().val()
        return 0 if index is None else int(index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching index: {str(e)}")

def query_parser(query):
    try:
        if '@' in query:
            return int(query.split('@')[1][-1])
        return 0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid query format: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello World from Mugundhan"}

@app.post("/generate")
def generate_response(request: ImageRequest):
    try:
        uid = request.id
        query = request.query
        flag = check_query(query)
        
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
            
            message = HumanMessage(content=[
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': path}
            ])
            put_index(uid, index)
            response = llm1.invoke([message])
        else:
            response = generate_answer(query)
        
        title, questions = generate_questions(response.content)
        put_context(uid, query, response.content)
        
        return {"title": title, "questions": questions, "response": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
