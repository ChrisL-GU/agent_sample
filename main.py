import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
context = []

tools = [
    {
        "type": "function",
        "name": "read_file",
        "description": "read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "name of the file to read",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "type": "function",
        "name": "count_tokens",
        "description": "count the number of tokens in a string",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "the content to count tokens in",
                },
            },
            "required": ["content"],
        },
    },
    {
        "type": "function",
        "name": "ping",
        "description": "ping some host on the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "hostname or IP",
                },
            },
            "required": ["host"],
        },
    },
]


def count_tokens(content):
    return str(len(content))


def read_file(filename="README.md"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"error: {e}"


def ping(host=""):
    try:
        result = subprocess.run(
            ["ping", "-c", "5", host],
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        return result.stdout
    except Exception as e:
        return f"error: {e}"


def call(tools):
    return client.responses.create(model="gpt-5", tools=tools, input=context)


def tool_call(item):
    tool_map = {
        "ping": ping,
        "read_file": read_file,
        "count_tokens": count_tokens,
    }
    args = json.loads(item.arguments)
    func = tool_map.get(item.name)
    if func:
        result = func(**args)
    else:
        result = f"error: unknown tool '{item.name}'"
    print(f"Tool call: {item.arguments}")
    return [
        item,
        {"type": "function_call_output", "call_id": item.call_id, "output": result},
    ]


def handle_tools(tools, response):
    if response.output[0].type == "reasoning":
        context.append(response.output[0])
    original_size = len(context)
    for item in response.output:
        if item.type == "function_call":
            context.extend(tool_call(item))
    return len(context) != original_size


def process(line):
    context.append({"role": "user", "content": line})
    response = call(tools)

    while handle_tools(tools, response):
        response = call(tools)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


def main():
    while True:
        line = input("> ")
        result = process(line)
        print(f">>> {result}\n")


if __name__ == "__main__":
    main()
