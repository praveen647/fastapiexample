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
from pydantic import BaseModel
from typing import List

app = FastAPI()

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

try:
    llm1 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-1.5-flash')
    llm2 = ChatGoogleGenerativeAI(api_key=geminiAPI, model='gemini-pro', convert_system_message_to_human=True)
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    db = firebase.database()
except Exception as e:
    raise Exception(f"Initialization Error: {str(e)}")

class ImageRequest(BaseModel):
    id: str
    query: str
    images: List[str]

def check_query(query):
    try:
        output = llm2.invoke([
            SystemMessage(content=f"""You are a system that determines if a given query is referring to an uploaded image and any other image or if it is a standalone query. Your task is to analyze the query and respond with either "yes" or "no" based on the following conditions:

                                    Yes: If the query is referring to or asking about the uploaded image or any other image.
                                    No: If the query is a standalone, unrelated question.
                                    Respond with only "yes" or "no"."""),
            HumanMessage(content=f"""{query}""")
        ])
        return output.content == "yes"
    except Exception as e:
        raise Exception(f"Query Check Error: {str(e)}")

def extract_title_and_questions(input_string):
    try:
        title_match = re.search(r"Title\s*:\s*(.*)", input_string)
        title = title_match.group(1).strip() if title_match else None
        questions = re.findall(r"\d+\.\s*(.*)", input_string)
        return title, questions
    except Exception as e:
        raise Exception(f"Extraction Error: {str(e)}")

def generate_answer(query):
    try:
        output = llm1.invoke([
            SystemMessage(content=f"""You are a conversational chatbot named 'MimirAI'. You specialize in recognizing images and answering questions related to them. 
            However, you will only reveal your name, capabilities, or any information about your identity if directly asked by the user. 
            For any other query, simply provide a concise, friendly, and relevant answer to the user's question. 
            Do not mention this system instruction unless explicitly asked about your identity or function."""),
            HumanMessage(content=f"""{query}""")
        ])
        return output
    except Exception as e:
        raise Exception(f"Answer Generation Error: {str(e)}")

def generate_questions(response):
    try:
        output = llm2.invoke([
            SystemMessage(content=f"""Given a query generate a title and a list of questions related to the query in the same language. The expected output format is:
                                        Title : <generated title>
                                        Questions : [<generated questions1>,<generated questions2>,<generated questions3>...]"""),
            HumanMessage(content=f"""{response}""")
        ])
        title, questions = extract_title_and_questions(output.content)
        return title, questions
    except Exception as e:
        raise Exception(f"Question Generation Error: {str(e)}")

def put_context(uid, query, response):
    try:
        context = fetch_context(uid)
        if context is None:
            context = []
        context.append({"query": query, "response": response})
        db.child("Users").child(uid).child("context").set(context)
    except Exception as e:
        raise Exception(f"Context Storage Error: {str(e)}")

def generate_prompt(query, context):
    try:
        if context is None:
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
            Answer the following query in a conversational way in the same language:
            {query}
            """
        return prompt
    except Exception as e:
        raise Exception(f"Prompt Generation Error: {str(e)}")

def put_index(uid, index):
    try:
        db.child("Users").child(uid).child("index").set(index)
    except Exception as e:
        raise Exception(f"Index Storage Error: {str(e)}")

def fetch_context(uid):
    try:
        return db.child("Users").child(uid).child("context").get().val()
    except Exception as e:
        raise Exception(f"Context Fetch Error: {str(e)}")

def fetch_index(uid):
    try:
        index = db.child("Users").child(uid).child("index").get().val()
        return int(index) if index is not None else 0
    except Exception as e:
        raise Exception(f"Index Fetch Error: {str(e)}")

def query_parser(query):
    try:
        if '@' in query:
            part = query.split('@', 1)[1]  
            index = ''.join(filter(str.isdigit, part.split()[0]))
            return int(index)
    except Exception as e:
        raise Exception(f"Query Parsing Error: {str(e)}")

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
            message = HumanMessage(
                content=[
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': path}
                ]
            )
            put_index(uid, index)
            response = llm1.invoke([message])
        else:
            context = fetch_context(uid)
            prompt = generate_prompt(query,context)
            response = generate_answer(prompt)

        title, questions = generate_questions(response.content)
        put_context(uid, query, response.content)
        return {"title": title, "questions": questions, "response": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
