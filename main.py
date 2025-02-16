import argparse
import json
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Optional, Dict, Any, Generator

import requests
from tqdm import tqdm


# Global variables for download tracking
total_bytes_downloaded = 0
download_lock = threading.Lock()

def validate_path(path: str, error_prefix: str) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{error_prefix} not found at: {path}")
    return path_obj

def clean_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def download_file(url: str, dest: Path) -> Tuple[bool, str]:
    global total_bytes_downloaded
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    with download_lock:
                        total_bytes_downloaded += len(chunk)
        
        return True, ""
    except Exception as e:
        return False, str(e)

def print_download_progress(stop_event: threading.Event, model_name: str):
    while not stop_event.is_set():
        with download_lock:
            gb = total_bytes_downloaded / (1024**3)
        sys.stdout.write(f"\rDownloading {model_name}: {gb:.3f}GB")
        sys.stdout.flush()
        time.sleep(2)
    print()  # New line after download completes

def download_model(model_url: str, temp_dir: Path, model_name: str) -> Tuple[Path, List[str]]:
    global total_bytes_downloaded
    total_bytes_downloaded = 0  # Reset counter for each new model
    
    # Start progress thread
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=print_download_progress, args=(stop_event, model_name))
    progress_thread.start()

    try:
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model_url)
        if not match:
            # Single file download
            dest = temp_dir / Path(model_url).name
            success, error = download_file(model_url, dest)
            if not success:
                raise RuntimeError(f"Download failed: {error}")
            return dest, [model_url]

        # Multi-part download
        base_url = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", model_url)
        total_parts = int(match.group(2))
        urls = [f"{base_url}-{i:05d}-of-{total_parts:05d}.gguf" 
                for i in range(1, total_parts + 1)]
        
        with ThreadPoolExecutor() as executor:
            futures = []
            files = []
            download_errors = []
            
            for url in urls:
                dest = temp_dir / Path(url).name
                files.append(dest)
                futures.append(executor.submit(download_file, url, dest))
            
            for future in futures:
                success, error = future.result()
                if not success:
                    download_errors.append(error)
            
            if download_errors:
                raise RuntimeError(
                    f"Failed to download {len(download_errors)} parts: {download_errors}"
                )

        return files[0], urls
    finally:
        stop_event.set()
        progress_thread.join()

def run_llm_inference(llama_run_path: Path, model_file: Path, prompt: str, context_len: str) -> str:
    process = subprocess.run(
        [str(llama_run_path), "-c", str(context_len), str(model_file), prompt],
        capture_output=True,
        text=True
    )
    if process.returncode == 0:
        output = process.stdout
        output = clean_ansi_escape_sequences(output)
        output = output.strip()
        return output
    else:
        raise Exception("llama-run did not exit with 0")

def process_model(
    model_config: dict,
    prompts: List[dict],
    llama_run_path: Path,
    context_len: int,
    verbose: bool,
    local_model_path: Optional[Path] = None
) -> Generator[Dict[str, Any], None, None]:
    model_name = model_config['name']
    
    if local_model_path:
        main_file = local_model_path
        with tqdm(prompts, desc=f"Processing {model_name}", leave=True) as pbar:
            for prompt in pbar:
                pbar.set_description(f"Model: {model_name} | Prompt: {prompt['name']}")
                response = run_llm_inference(
                    llama_run_path, main_file, prompt["prompt"], context_len
                )
                result_entry = {
                    "model_name": model_name,
                    "prompt_name": prompt["name"],
                    "prompt": prompt["prompt"],
                    "response": response
                }
                yield result_entry
                if verbose:
                    tqdm.write(f"Processed {model_name} - {prompt['name']}: {response[:50]}...")
        print()  # New line after processing all prompts
    else:
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            try:
                main_file, _ = download_model(model_config["url"], temp_dir, model_name)
            except Exception as e:
                print(f"\nFailed to download model {model_name}: {str(e)}")
                return
            
            with tqdm(prompts, desc=f"Processing {model_name}", leave=True) as pbar:
                for prompt in pbar:
                    pbar.set_description(f"Model: {model_name} | Prompt: {prompt['name']}")
                    response = run_llm_inference(
                        llama_run_path, main_file, prompt["prompt"], context_len
                    )
                    result_entry = {
                        "model_name": model_name,
                        "prompt_name": prompt["name"],
                        "prompt": prompt["prompt"],
                        "response": response
                    }
                    yield result_entry
                    if verbose:
                        tqdm.write(f"Processed {model_name} - {prompt['name']}: {response[:50]}...")
            print()  # New line after processing all prompts

def main():
    parser = argparse.ArgumentParser(description='LLM Benchmarking Tool')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--llama-run-path', required=True, help='Path to llama-run')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--local-model', help='Path to local model file (bypasses downloads)')
    
    args = parser.parse_args()

    # Validate paths
    config_path = validate_path(args.config, "Config file")
    llama_run_path = validate_path(args.llama_run_path, "llama-run executable")
    output_path = Path(args.output)
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    settings = config["settings"]
    
    # Prepare output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results or initialize
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []
    else:
        all_results = []

    if args.local_model:
        local_model_path = validate_path(args.local_model, "Local model file")
        model_name = local_model_path.stem
        # Filter unprocessed prompts
        processed_prompts = {res['prompt_name'] for res in all_results if res['model_name'] == model_name}
        unprocessed_prompts = [p for p in config["prompts"] if p['name'] not in processed_prompts]
        
        if not unprocessed_prompts:
            print(f"All prompts for local model {model_name} are already processed.")
            return
        
        model_generator = process_model(
            model_config={"name": model_name, "url": ""},
            prompts=unprocessed_prompts,
            llama_run_path=llama_run_path,
            context_len=settings["context_len"],
            verbose=args.verbose,
            local_model_path=local_model_path
        )
        
        for result in model_generator:
            all_results.append(result)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=4)
                f.write('\n')
    else:
        for model_config in config["models"]:
            model_name = model_config['name']
            # Filter unprocessed prompts
            processed_prompts = {res['prompt_name'] for res in all_results if res['model_name'] == model_name}
            unprocessed_prompts = [p for p in config["prompts"] if p['name'] not in processed_prompts]
            
            if not unprocessed_prompts:
                print(f"Skipping model {model_name} (all prompts processed)")
                continue
            
            model_generator = process_model(
                model_config=model_config,
                prompts=unprocessed_prompts,
                llama_run_path=llama_run_path,
                context_len=settings["context_len"],
                verbose=args.verbose,
                local_model_path=None
            )
            
            for result in model_generator:
                all_results.append(result)
                with open(output_path, 'w') as f:
                    json.dump(all_results, f, indent=4)
                    f.write('\n')

if __name__ == "__main__":
    main()

