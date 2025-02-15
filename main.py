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
from typing import List, Tuple

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

def print_download_progress(stop_event: threading.Event):
    while not stop_event.is_set():
        with download_lock:
            gb = total_bytes_downloaded / (1024**3)
        sys.stdout.write(f"\rDownloaded: {gb:.3f} GB")
        sys.stdout.flush()
        time.sleep(2)
    print()  # New line after download completes

def download_model(model_url: str, temp_dir: Path) -> Tuple[Path, List[str]]:
    global total_bytes_downloaded
    total_bytes_downloaded = 0  # Reset counter for each new model
    
    # Start progress thread
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=print_download_progress, args=(stop_event,))
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

def run_llm_inference(llama_run_path: Path, model_file: Path, prompt: str) -> str:
    process = subprocess.run(
        [str(llama_run_path), str(model_file), prompt],
        capture_output=True,
        text=True
    )
    return process.stdout.strip() if process.returncode == 0 else ""

def process_model(
    model_config: dict,
    prompts: List[dict],
    llama_run_path: Path,
    output_path: Path,
    verbose: bool
):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        try:
            main_file, downloaded_urls = download_model(
                model_config["url"], temp_dir
            )
        except Exception as e:
            print(f"\nFailed to download model {model_config['name']}: {str(e)}")
            return
        
        model_name = model_config['name']
        
        with tqdm(prompts, desc=f"Processing {model_name}", leave=False) as pbar:
            for prompt in pbar:
                result_entry = {
                    "model_name": model_name,
                    "prompt_name": prompt["name"],
                    "prompt": prompt["prompt"],
                    "response": run_llm_inference(
                        llama_run_path, main_file, prompt["prompt"]
                    )
                }
                
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result_entry) + "\n")
                
                if verbose:
                    tqdm.write(
                        f"Processed {model_name} - {prompt['name']}: "
                        f"{result_entry['response'][:50]}..."
                    )

def main():
    parser = argparse.ArgumentParser(description='LLM Benchmarking Tool')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--llama_run_path', required=True, help='Path to llama-run')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    # Validate paths
    config_path = validate_path(args.config, "Config file")
    llama_run_path = validate_path(args.llama_run_path, "llama-run executable")
    output_path = Path(args.output)
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w'): pass  # Truncate file
    
    # Process models
    for model_config in tqdm(config["models"], desc="Models"):
        process_model(
            model_config=model_config,
            prompts=config["prompts"],
            llama_run_path=llama_run_path,
            output_path=output_path,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()

