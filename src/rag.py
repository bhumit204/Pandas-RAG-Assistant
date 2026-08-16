from vector_db import search
from dotenv import load_dotenv
from openai import OpenAI
import json


search_tool = {
    "type": "function",
    "name": "search",
    "description": (
        "Search the Pandas documentation stored in the vector database. "
        "Returns the most relevant documentation chunks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

instructions = """You are a Pandas documentation assistant.

Always answer only from the retrieved documentation.

For every user question:

1. Search the vector database.
2. If the retrieved context is insufficient,
   reformulate the query and search again.
3. You may perform multiple searches.
4. Never invent information.
5. If the answer is not found after several searches,
   reply that the documentation does not contain the answer."""

TOOLS = {
    "search": search
}

def make_call(call):
    args = json.loads(call.arguments)
    result = TOOLS[call.name](**args)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(result, indent=2),
    }

load_dotenv()
openai_client = OpenAI()

def agentic_rag(query):
    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "developer",
                "content": instructions,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        tools=[search_tool],
    )

    MAX_ITERATIONS = 3
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        tool_outputs = []

        for item in response.output:

            if item.type == "function_call":

                tool_outputs.append(
                    make_call(item)
                )

            elif item.type == "message":
                return item.content[0].text

        if not tool_outputs:
            break

        response = openai_client.responses.create(
            model="gpt-5.4-mini",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=[search_tool],
        )
