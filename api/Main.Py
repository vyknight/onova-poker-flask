# flask
from flask import Flask, jsonify, request
import json
# langchain imports
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pinecone
import os
from decouple import config

# api keys
os.environ["OPENAI_API_KEY"] = config('OPENAI_API')
os.environ["GOOGLE_API_KEY"] = config('GOOGLE_API')
os.environ["GOOGLE_CSE_ID"] = config('GOOGLE_CSE')

pinecone.init(
    api_key=config('PINECONE_API'),
    environment="us-west4-gcp-free"
)

# llm
docsearch_llm = OpenAI(temperature=0.1)

# vector store
embeddings = OpenAIEmbeddings()
index_name = "poker-gpt"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
qa = RetrievalQA.from_chain_type(llm=docsearch_llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 5}))

# agent building
# pot odds tool
def parse_potodds(input):
  opponent_bet, pot_size = input.split(",")
  return calculatePotOdds(float(opponent_bet), float(pot_size))

def calculatePotOdds(opponent_bet: float, pot_size: float):
  return opponent_bet / (opponent_bet + pot_size)

pot_odds_tool = Tool(
    name = "Pot Odds Tool",
    func=parse_potodds,
    description=(
      "use this tool when you need to calculate the pot odds in a poker game"
      "given both the current pot size and the amount that the opponent has bet"
      "To use the tool, you must provide both of the following parameters in a single string"
      "`opponent_bet, pot_size`."
    )
)

qa_tool = Tool(
        name="Poker Textbooks",
        func=qa.run,
        description="""
            use this as the primary source of context information when you are asked the question.
            Always search for the answers using this tool right after web searching for the odds, 
            don't make up answers yourself
        """,
)

tools = [qa_tool, pot_odds_tool] + load_tools(["llm-math", "google-search"], llm=docsearch_llm)  # minor type issue

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_llm = ChatOpenAI(temperature=0.1)

agent_chain = initialize_agent(tools, agent_llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory,
                               verbose=False)   # currently verbose off to test output

sys = """
Agent is a large language model trained by OpenAI.
Agent is able to assist with a wide range of tasks, from answering simple questions to providing in depth explanations and discussions on a wide range of topics. 
However, this specific agent is designed to act in the role of a Poker coach, offering good advice for beginner to intermediate players of Texas Hold'em pokers. 
Agent is able to generate human like text based on the input it receives, allowing it to engage in natural sounding conversations and provide coherent and relevant responses to the topic at hand.

Agent is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand a large amount of text, and can use this knowledge to provide
accurate and informative responses to a wide range of questions. This is especially true regarding the large amount of poker text the agent has received as a part of its training. The agent will seek to use
the poker texts it has trained on to provide the most accurate and helpful poker advice. 

When given a Poker hand, the Agent will first attempt to analyze the winning odds of that hand, as well as the pot odds through a google web search, before offering a recommendation on what action should be taken by the user. Unfortunately,
the agent is unable to calculate poker odds, as such it will always seek to use its tools, such as a web search, to calculate these odds. Afterwards, the agent will consult its poker books tool for analysis. 

Overall, the Agent is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific poker hand or just want to 
have a conversation, Agent is here to assist.

"""

new_prompt = agent_chain.agent.create_prompt(
    system_message=sys,
    tools = tools
)

agent_chain.agent.llm_chain.prompt = new_prompt


# flask setup
app = Flask(__name__)


@app.route('/')
def landing():
    return "Welcome to the API"


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
    app.run(host='0.0.0.0', port=5000, debug=True)

