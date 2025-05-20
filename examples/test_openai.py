"""
Simple test of OpenAI client to diagnose proxies issue.
"""

import os
from dotenv import load_dotenv
from cdas.ai.openai_wrapper import create_openai_client

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    """Test basic OpenAI client functionality."""
    print(f"Testing OpenAI client with key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}")
    
    # Initialize client with verbose debugging
    print(f"Creating OpenAI client with these parameters:")
    print(f"API Key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}")
    print(f"Environment proxies: {os.environ.get('HTTP_PROXY')}, {os.environ.get('HTTPS_PROXY')}")
    
    # Use our custom wrapper to create the client
    client = create_openai_client(api_key=OPENAI_API_KEY)
    
    # Make a simple request
    try:
        # Try with a more commonly available model
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ]
        )
        
        print("Response received:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    main()