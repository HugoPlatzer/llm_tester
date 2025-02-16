import argparse
import json
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Optional, Dict, Any, Generator, Callable, Set
import requests
from tqdm import tqdm


class DownloadProgressTracker:
    """Tracks download progress and displays updates in a background thread."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_bytes = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._print_progress)

    def _print_progress(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                gb = self.total_bytes / (1024 ** 3)
            sys.stdout.write(f"\rDownloading {self.model_name}: {gb:.3f}GB")
            sys.stdout.flush()
            time.sleep(2)
        print()

    def __enter__(self) -> 'DownloadProgressTracker':
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_event.set()
        self.thread.join()

    def update(self, chunk_size: int) -> None:
        with self.lock:
            self.total_bytes += chunk_size


def validate_path(path: str, error_prefix: str) -> Path:
    """Validates and returns a Path object if it exists."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{error_prefix} not found at: {path}")
    return path_obj


def clean_ansi_escape_sequences(text: str) -> str:
    """Removes ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def download_file(
    url: str,
    dest: Path,
    on_chunk: Optional[Callable[[int], None]] = None
) -> Tuple[bool, str]:
    """Downloads a file with optional progress tracking."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if on_chunk:
                        on_chunk(len(chunk))
        return True, ""
    except Exception as e:
        return False, str(e)


def download_model_part(
    base_url: str,
    part_number: int,
    total_parts: int,
    dest_dir: Path,
    on_chunk: Callable[[int], None]
) -> Tuple[Path, str]:
    """Downloads a single part of a multi-part model."""
    url = f"{base_url}-{part_number:05d}-of-{total_parts:05d}.gguf"
    dest = dest_dir / Path(url).name
    success, error = download_file(url, dest, on_chunk)
    if not success:
        raise RuntimeError(f"Failed to download {url}: {error}")
    return dest, url


def download_model(
    model_url: str,
    dest_dir: Path,
    model_name: str
) -> Tuple[Path, List[str]]:
    """Handles both single-file and multi-part model downloads."""
    with DownloadProgressTracker(model_name) as tracker:
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model_url)
        if not match:
            dest = dest_dir / Path(model_url).name
            success, error = download_file(model_url, dest, tracker.update)
            if not success:
                raise RuntimeError(f"Download failed: {error}")
            return dest, [model_url]

        total_parts = int(match.group(2))
        base_url = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", model_url)
        urls = [f"{base_url}-{i:05d}-of-{total_parts:05d}.gguf" 
                for i in range(1, total_parts + 1)]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    download_model_part,
                    base_url,
                    i,
                    total_parts,
                    dest_dir,
                    tracker.update
                )
                for i in range(1, total_parts + 1)
            ]
            
            results = []
            for future in futures:
                results.append(future.result())

        return results[0][0], urls


def run_inference(
    llama_run_path: Path,
    model_file: Path,
    prompt: str,
    context_len: int
) -> str:
    """Runs the LLM inference and returns cleaned output."""
    process = subprocess.run(
        [str(llama_run_path), "-c", str(context_len), str(model_file), prompt],
        capture_output=True,
        text=True
    )
    if process.returncode != 0:
        raise RuntimeError("llama-run execution failed")
    
    output = clean_ansi_escape_sequences(process.stdout).strip()
    return output


@contextmanager
def get_model_file(
    model_config: Dict[str, Any],
    local_model_path: Optional[Path]
) -> Generator[Tuple[Path, str], None, None]:
    """Context manager for model file acquisition (local or download)."""
    model_name = model_config['name']
    
    if local_model_path:
        yield local_model_path, model_name
        return

    with TemporaryDirectory() as temp_dir:
        dest_dir = Path(temp_dir)
        try:
            main_file, _ = download_model(
                model_config["url"],
                dest_dir,
                model_name
            )
            yield main_file, model_name
        except Exception as e:
            print(f"\nFailed to download model {model_name}: {str(e)}")
            raise


def process_prompts(
    model_file: Path,
    model_name: str,
    prompts: List[Dict[str, str]],
    llama_run_path: Path,
    context_len: int,
    verbose: bool
) -> Generator[Dict[str, Any], None, None]:
    """Processes all prompts through the model and yields results."""
    with tqdm(prompts, desc=f"Processing {model_name}", leave=True) as pbar:
        for prompt in pbar:
            pbar.set_description(f"Model: {model_name} | Prompt: {prompt['name']}")
            response = run_inference(
                llama_run_path,
                model_file,
                prompt["prompt"],
                context_len
            )
            result = {
                "model_name": model_name,
                "prompt_name": prompt["name"],
                "prompt": prompt["prompt"],
                "response": response
            }
            if verbose:
                tqdm.write(f"Processed {model_name} - {prompt['name']}: {response[:50]}...")
            yield result


def get_unprocessed_prompts(
    model_name: str,
    all_prompts: List[Dict[str, str]],
    existing_results: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Filters out already processed prompts for a model."""
    processed = {
        res['prompt_name']
        for res in existing_results
        if res['model_name'] == model_name
    }
    return [p for p in all_prompts if p['name'] not in processed]


def write_results(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """Writes results to JSON file with indentation."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        f.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM Benchmarking Tool')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--llama-run-path', required=True, help='Path to llama-run')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--local-model', help='Path to local model file')
    args = parser.parse_args()

    # Validate and setup paths
    config_path = validate_path(args.config, "Config file")
    llama_run_path = validate_path(args.llama_run_path, "llama-run executable")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    settings = config["settings"]

    # Initialize or load results
    try:
        existing_results = json.loads(output_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []
    all_results = existing_results.copy()

    # Determine models to process
    models_to_process = []
    if args.local_model:
        local_model_path = validate_path(args.local_model, "Local model file")
        model_name = local_model_path.stem
        models_to_process.append({
            "name": model_name,
            "url": None
        })
    else:
        models_to_process = config["models"]

    # Process each model
    for model_config in models_to_process:
        model_name = model_config["name"]
        unprocessed = get_unprocessed_prompts(
            model_name,
            config["prompts"],
            existing_results
        )
        if not unprocessed:
            print(f"Skipping {model_name} (all prompts processed)")
            continue

        try:
            with get_model_file(
                model_config,
                local_model_path=Path(args.local_model) if args.local_model else None
            ) as (model_file, _):
                for result in process_prompts(
                    model_file,
                    model_name,
                    unprocessed,
                    llama_run_path,
                    settings["context_len"],
                    args.verbose
                ):
                    all_results.append(result)
                    write_results(output_path, all_results)

        except Exception as e:
            print(f"\nError processing {model_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()

