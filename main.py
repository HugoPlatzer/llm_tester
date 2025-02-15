import argparse
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys

def validate_path(path: str, error_prefix: str) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{error_prefix} not found at: {path}")
    return path_obj

def run_llm_inference(llama_exec: Path, model_file: Path, prompt: str) -> tuple[str, str]:
    """Run LLM inference on a single prompt and return (response, error) tuple."""
    process = subprocess.run(
        [str(llama_exec), str(model_file), prompt],
        capture_output=True,
        text=True
    )
    
    if process.returncode != 0:
        return "", process.stderr.strip()
    
    return process.stdout.strip(), ""

def process_prompts(llama_exec: Path, model_file: Path, prompts: list, output_path: Path, verbose: bool):
    """Process all prompts and save results incrementally."""
    results = []
    
    with tqdm(prompts, desc="Processing prompts", file=sys.stdout) as pbar:
        for entry in pbar:
            if verbose:
                tqdm.write(f"\nProcessing prompt '{entry['name']}': {entry['prompt']}")
            else:
                pbar.set_postfix_str(f"Processing: {entry['name']}")

            response, error = run_llm_inference(llama_exec, model_file, entry['prompt'])
            
            if error:
                response = ""
                if verbose:
                    tqdm.write(f"Error: {error}")

            result_entry = {
                "name": entry["name"],
                "prompt": entry["prompt"],
                "response": response
            }
            results.append(result_entry)

            # Write after each completion
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)

    return results

def main():
    parser = argparse.ArgumentParser(description='Run LLM on prompts using llama.cpp')
    parser.add_argument('--llama_run_path', required=True,
                      help='Path to llama-run executable')
    parser.add_argument('--model', required=True,
                      help='Path to GGUF model file')
    parser.add_argument('--input', required=True,
                      help='Input JSON file with prompts')
    parser.add_argument('--output', required=True,
                      help='Output JSON file to save responses')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed processing information')
    
    args = parser.parse_args()

    # Validate paths
    llama_exec = validate_path(args.llama_run_path, "llama-run executable")
    model_file = validate_path(args.model, "Model file")
    input_file = validate_path(args.input, "Input prompts file")
    output_path = Path(args.output)

    # Read prompts
    with open(input_file, 'r') as f:
        prompts = json.load(f)

    # Process prompts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = process_prompts(llama_exec, model_file, prompts, output_path, args.verbose)

    print(f"\nCompleted processing {len(prompts)} prompts. Final results saved to: {output_path}")

if __name__ == "__main__":
    main()

