import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import json
import re

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

app = Flask(__name__)

# Initialize session memory
session_memory = {}

# Available tools for the agent
def web_search(query, max_results=5):
    """Search the web using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def calculator(expression):
    """Evaluate a mathematical expression safely"""
    try:
        # Remove any potentially dangerous characters
        expression = re.sub(r'[^0-9+\-*/().]', '', expression)
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

# Available tools
TOOLS = {
    "web_search": web_search,
    "calculator": calculator
}

TOOL_DESCRIPTIONS = [
    {
        "name": "web_search",
        "description": "Search the web for current information. Input should be a search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "number",
                    "description": "Number of results to return, default is 5",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Input should be a valid expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]

def generate_response(prompt, session_id, max_iterations=5):
    """Generate a response using Gemini with tool usage"""
    if session_id not in session_memory:
        session_memory[session_id] = []
    
    # Add conversation history to prompt
    history = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                         for msg in session_memory[session_id][-5:]])  # Last 5 exchanges
    full_prompt = f"Previous conversation:\n{history}\n\nCurrent query: {prompt}"
    
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Start with the initial prompt
    current_prompt = f"""You are an AI assistant with access to tools. 
    You can use these tools to help answer questions. 
    Available tools: {json.dumps(TOOL_DESCRIPTIONS)}
    
    If you need to use a tool, respond with exactly:
    ACTION: tool_name
    ACTION INPUT: JSON parameters
    
    After taking action, you will receive the result and can continue.
    
    Current conversation:
    {full_prompt}
    
    Please provide a helpful response or specify an action to take:"""
    
    for _ in range(max_iterations):
        response = model.generate_content(current_prompt)
        response_text = response.text.strip()
        
        # Check if the model wants to use a tool
        if response_text.startswith("ACTION:"):
            try:
                # Parse the action
                lines = response_text.split('\n')
                action_line = lines[0]
                input_line = lines[1] if len(lines) > 1 else ""
                
                if "ACTION:" in action_line and "ACTION INPUT:" in input_line:
                    tool_name = action_line.split("ACTION:")[1].strip()
                    input_json = input_line.split("ACTION INPUT:")[1].strip()
                    
                    # Parse the JSON input
                    try:
                        params = json.loads(input_json)
                    except:
                        # If JSON parsing fails, try to extract key-value pairs
                        params = {}
                        if "query" in input_json.lower():
                            params["query"] = input_json
                        elif "expression" in input_json.lower():
                            params["expression"] = input_json
                    
                    # Execute the tool
                    if tool_name in TOOLS:
                        result = TOOLS[tool_name](**params)
                        current_prompt = f"Tool {tool_name} returned: {result}\n\nHow would you like to proceed with this information?"
                    else:
                        current_prompt = f"Tool {tool_name} not found. Available tools: {list(TOOLS.keys())}"
                else:
                    current_prompt = "Invalid action format. Please use exactly: ACTION: tool_name\nACTION INPUT: JSON parameters"
            
            except Exception as e:
                current_prompt = f"Error executing action: {str(e)}. Please try again."
        else:
            # No action needed, return the response
            session_memory[session_id].append({
                "user": prompt,
                "assistant": response_text
            })
            return response_text
    
    # If we've reached max iterations without a final response
    return "I've reached the maximum number of iterations. Please try a simpler query."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = generate_response(message, session_id)
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in session_memory:
            session_memory[session_id] = []
        
        return jsonify({'status': 'memory cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)