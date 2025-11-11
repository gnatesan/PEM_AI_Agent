# agent/agent_runner.py

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Import your custom tools
from tools.vector_search import VectorSearchTool
from tools.email_tool import EmailTool
# from tools.calendar_tool import CalendarTool  # Optional


from dotenv import load_dotenv
load_dotenv()


def load_tools():
    """Wrap all tools into LangChain Tool objects."""
    vector_tool = VectorSearchTool()
    email_tool = EmailTool()
    # calendar_tool = CalendarTool()

    return [
        Tool.from_function(
            func=vector_tool._run,
            name=vector_tool.name,
            description=vector_tool.description,
        ),
        Tool.from_function(
            func=email_tool._run,
            name=email_tool.name,
            description=email_tool.description,
        ),
        # Add more tools here as needed
    ]

def initialize_langchain_agent():
    """Initialize the LangChain agent with registered tools and LLM."""
    tools = load_tools()

    # Use OpenAI's function-calling LLM (e.g. gpt-4 or gpt-3.5-turbo)
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    return agent

def main():
    agent = initialize_langchain_agent()
    query = "What are the signs of diabetic ketoacidosis in children?"
    response = agent.run(query)
    print("\n💬 Agent Response:\n", response)

if __name__ == "__main__":
    main()

