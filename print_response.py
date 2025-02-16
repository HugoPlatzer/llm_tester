import argparse
import json
from termcolor import colored

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Find model responses')
    parser.add_argument('json_file', help='Path to JSON data file')
    parser.add_argument('model_name', help='Name of the model')
    parser.add_argument('prompt_name', help='Name of the prompt')
    args = parser.parse_args()

    try:
        # Load JSON data from file
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(colored(f"Error: File '{args.json_file}' not found", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Invalid JSON in file '{args.json_file}'", 'red'))
        return

    # Find matching entry
    result = None
    for entry in data:
        if entry['model_name'] == args.model_name and entry['prompt_name'] == args.prompt_name:
            result = entry
            break

    if not result:
        print(colored(f"No results found for model '{args.model_name}' and prompt '{args.prompt_name}'", 'red'))
        return

    # Print results with colors
    print(colored(f"Model:    {result['model_name']}", 'blue'))
    print(colored(f"Prompt:   {result['prompt_name']}", 'green'))
    print(colored(f"Question: {result['prompt']}", 'yellow'))
    print(f"Response: {result['response']}")

if __name__ == "__main__":
    main()

