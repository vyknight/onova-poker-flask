# flask
from flask import Flask, jsonify, request
import json

# api keys
import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GOOGLE_CSE_ID"] = ""

import pinecone
pinecone.init(
    api_key="",
    environment="us-west4-gcp-free"
)

# langchain imports
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# llm
docsearch_llm = OpenAI(temperature=0.1)

# vector store
embeddings = OpenAIEmbeddings()
index_name = "poker-gpt"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
qa = RetrievalQA.from_chain_type(llm=docsearch_llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 5}))

# agent building
tools = [
    Tool(
        name="Poker Textbooks",
        func=qa.run,
        description="""
            use this as the primary source of context information when you are asked the question.
            Always search for the answers using this tool right after web searching for the odds, 
            don't make up answers yourself
        """,
    )
] + load_tools(["llm-math", "google-search"], llm=docsearch_llm)  # minor type issue

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_llm = ChatOpenAI(temperature=0.1)

agent_chain = initialize_agent(tools, agent_llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory,
                               verbose=False)   # currently verbose off to test output

# flask setup
app = Flask(__name__)


@app.route('/response', methods=['POST'])
def get_agent_response():
    print("request received")
    json_obj = request.get_json()
    temp = json.dumps(json_obj)
    req_dic = json.loads(temp)
    print(req_dic)
    prompt = req_dic["prompt"]
    print(prompt)
    response = agent_chain.run(prompt)
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run()

