"""
Absolutely minimal OpenAI test client.
"""

import os
from dotenv import load_dotenv
import inspect

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    """Test basic OpenAI client functionality."""
    print(f"Testing OpenAI client with key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}")
    
    # Import OpenAI with special debugging
    from openai import OpenAI
    
    # Print the arguments that OpenAI's __init__ expects
    init_sig = inspect.signature(OpenAI.__init__)
    print("OpenAI.__init__ expected arguments:")
    for param_name, param in init_sig.parameters.items():
        print(f"  {param_name}: {param.annotation}")
    
    # Try to create a client with only the minimal params
    try:
        print("Attempting to create client with only api_key...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("Success! Client created.")
        
        # Test a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello world"}
            ]
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        
        # If error is related to proxies, try to debug
        if "proxies" in str(e):
            print("\nDebugging proxies issue:")
            import httpx
            print("HTTPX client parameters:")
            httpx_sig = inspect.signature(httpx.Client.__init__)
            for param_name, param in httpx_sig.parameters.items():
                print(f"  {param_name}: {param.annotation}")

if __name__ == "__main__":
    main()