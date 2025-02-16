import sys
import json

# Read multi-line input until EOF (Ctrl+D on Unix, Ctrl+Z on Windows)
input_text = sys.stdin.read()

# Convert to JSON string with proper escaping
json_string = json.dumps(input_text)

# Print the result
print(json_string)
