import argparse
import json
import re
import subprocess
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Optional, Dict, Any, Generator, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    on_chunk: Optional[Callable[[int], None]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
    timeout: Tuple[int, int] = (30, 60),
    chunk_size: int = 65536
) -> Tuple[bool, str]:
    """Downloads a file with retries, exponential backoff, and progress tracking."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=retry_delay,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Track content length for validation
            content_length = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_chunk:
                            on_chunk(len(chunk))
            
            # Validate download size if Content-Length was provided
            if content_length > 0 and downloaded != content_length:
                raise requests.exceptions.ContentDecodingError(
                    f"Incomplete download (expected {content_length}, got {downloaded})"
                )
                
            return True, ""
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                current_delay = retry_delay * (2 ** attempt)
                print(f"Retry {attempt+1}/{max_retries} for {url} after error: {str(e)[:100]}...")
                time.sleep(current_delay)
            else:
                return False, f"Failed after {max_retries+1} attempts: {str(e)}"
        finally:
            session.close()
    
    return False, "Max retries exceeded"


def download_model_part(
    base_url: str,
    part_number: int,
    total_parts: int,
    dest_dir: Path,
    on_chunk: Callable[[int], None],
    max_retries: int,
    retry_delay: int,
    timeout: Tuple[int, int],
    chunk_size: int
) -> Tuple[Path, str]:
    """Downloads a single part of a multi-part model with enhanced robustness."""
    url = f"{base_url}-{part_number:05d}-of-{total_parts:05d}.gguf"
    dest = dest_dir / Path(url).name
    
    # Check for existing partial download
    if dest.exists():
        dest.unlink()
    
    success, error = download_file(
        url,
        dest,
        on_chunk,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        chunk_size=chunk_size
    )
    
    if not success:
        raise RuntimeError(f"Failed to download {url}: {error}")
    
    return dest, url


def download_model(
    model_url: str,
    dest_dir: Path,
    model_name: str,
    max_retries: int = 3,
    retry_delay: int = 5,
    timeout: Tuple[int, int] = (30, 60),
    max_workers: int = 4,
    chunk_size: int = 65536
) -> Tuple[Path, List[str]]:
    """Handles model downloads with improved parallelization and error handling."""
    with DownloadProgressTracker(model_name) as tracker:
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model_url)
        if not match:
            dest = dest_dir / Path(model_url).name
            success, error = download_file(
                model_url,
                dest,
                tracker.update,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                chunk_size=chunk_size
            )
            if not success:
                raise RuntimeError(f"Download failed: {error}")
            return dest, [model_url]

        total_parts = int(match.group(2))
        base_url = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", model_url)
        
        part_queue = queue.Queue()
        for i in range(1, total_parts + 1):
            part_queue.put(i)

        results = []
        failed = False
        lock = threading.Lock()

        def worker():
            nonlocal failed
            while not part_queue.empty() and not failed:
                try:
                    part_num = part_queue.get_nowait()
                    dest, url = download_model_part(
                        base_url,
                        part_num,
                        total_parts,
                        dest_dir,
                        tracker.update,
                        max_retries,
                        retry_delay,
                        timeout,
                        chunk_size
                    )
                    with lock:
                        results.append((dest, url))
                except Exception as e:
                    with lock:
                        failed = True
                    print(f"\nCritical error in part {part_num}: {str(e)}")
                    part_queue.queue.clear()
                finally:
                    part_queue.task_done()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(max_workers):
                executor.submit(worker)
            
            while not part_queue.empty() and not failed:
                time.sleep(1)
                remaining = part_queue.qsize()
                sys.stdout.write(f"\rParts remaining: {remaining}/{total_parts}")
                sys.stdout.flush()

        if failed:
            raise RuntimeError("Aborted due to download failure")

        # Extract part numbers and sort files correctly
        def get_part_number(path: Path) -> int:
            match = re.search(r"-(\d{5})-of-\d{5}\.gguf$", path.name)
            if not match:
                raise ValueError(f"Invalid part filename: {path.name}")
            return int(match.group(1))
        
        # Sort parts numerically and verify completeness
        downloaded_files = [dest for dest, _ in results]
        downloaded_files.sort(key=get_part_number)
        
        expected_parts = set(range(1, total_parts + 1))
        actual_parts = {get_part_number(f) for f in downloaded_files}
        
        if expected_parts != actual_parts:
            missing = expected_parts - actual_parts
            raise RuntimeError(f"Missing model parts: {missing}")

        # Return first part's path for inference
        main_file = downloaded_files[0]
        return main_file, [url for _, url in results]


def run_inference(
    llama_run_path: Path,
    model_file: Path,
    prompt: str,
    context_len: int,
    use_cuda: bool
) -> str:
    """Runs the LLM inference and returns cleaned output."""
    command = [str(llama_run_path), "-c", str(context_len)]
    if use_cuda:
        command.extend(["-ngl", "1000"])
    command.extend([str(model_file), prompt])
    
    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if process.returncode != 0:
        # Clean and capture debug information
        stderr_clean = clean_ansi_escape_sequences(process.stderr).strip()
        stdout_clean = clean_ansi_escape_sequences(process.stdout).strip()
        error_msg = (
            f"llama-run execution failed (exit code {process.returncode})\n"
            f"STDERR OUTPUT:\n{stderr_clean}\n"
            f"STDOUT OUTPUT:\n{stdout_clean}"
        )
        raise RuntimeError(error_msg)
    
    output = clean_ansi_escape_sequences(process.stdout).strip()
    return output


@contextmanager
def get_model_file(
    model_config: Dict[str, Any],
    local_model_path: Optional[Path],
    download_settings: Dict[str, Any]
) -> Generator[Tuple[Path, str], None, None]:
    """Context manager for model file acquisition with download settings."""
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
                model_name,
                **download_settings
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
    verbose: bool,
    use_cuda: bool
) -> Generator[Dict[str, Any], None, None]:
    """Processes all prompts through the model and yields results."""
    with tqdm(prompts, desc=f"Processing {model_name}", leave=True) as pbar:
        for prompt in pbar:
            pbar.set_description(f"Model: {model_name} | Prompt: {prompt['name']}")
            response = run_inference(
                llama_run_path,
                model_file,
                prompt["prompt"],
                context_len,
                use_cuda
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
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA offload with -ngl 1000')
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
    
    # Get download settings with defaults
    download_settings = config["settings"].get("download", {})
    download_settings.setdefault("max_retries", 3)
    download_settings.setdefault("retry_delay", 5)
    download_settings.setdefault("timeout", (30, 60))
    download_settings.setdefault("max_workers", 4)
    download_settings.setdefault("chunk_size", 65536)

    # ensure timeout setting is a tuple
    download_settings["timeout"] = tuple(download_settings["timeout"])

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
                local_model_path=Path(args.local_model) if args.local_model else None,
                download_settings=download_settings
            ) as (model_file, _):
                for result in process_prompts(
                    model_file,
                    model_name,
                    unprocessed,
                    llama_run_path,
                    config["settings"]["context_len"],
                    args.verbose,
                    args.cuda
                ):
                    all_results.append(result)
                    write_results(output_path, all_results)

        except Exception as e:
            print(f"\nError processing {model_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()

