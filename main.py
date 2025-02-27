import subprocess
import json
import argparse
import shutil
import os
import re


def get_directory_name(url):
    basename = os.path.basename(url)
    assert basename.endswith(".gguf")
    return basename.rstrip(".gguf")


def get_filename(url):
    return os.path.basename(url)


def get_urls_of_parts(url):
    pattern = r"(.*)(00001)-of-(\d{5}).gguf$" 
    match = re.match(pattern, url)
    if match is None:
        return [url]
    else:
        num_parts = int(match.group(3))
        urls = [url]
        for i in range(2, num_parts + 1):
            part_url = re.sub(pattern, f"\\g<1>{i:05d}-of-\\g<3>.gguf", url)
            urls.append(part_url)
        return urls


def download_model(url):
    aria2_cmd = "aria2c"
    num_parallel_connections = 3

    model_dir = get_directory_name(url)
    remove_downloaded_model(url)
    os.mkdir(model_dir)
    urls = get_urls_of_parts(url)
    for url in urls:
        cmd = [aria2_cmd,
                "-x", str(num_parallel_connections),
                "-d", model_dir,
                "-o", get_filename(url),
                url]
        subprocess.run(cmd, check=True)
    first_part_file = get_filename(urls[0])
    first_part_path = os.path.join(model_dir, first_part_file)
    return first_part_path


def remove_downloaded_model(url):
    model_dir = get_directory_name(url)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)


def get_prompts_for_model(config, results_list, model_name):
    results_for_model = [r for r in results_list if r["model_name"] == model_name]
    results_prompt_names = [r["prompt_name"] for r in results_for_model]
    all_prompt_names = [p["name"] for p in config["prompts"]]
    missing_prompt_names = [p for p in all_prompt_names if p not in results_prompt_names]
    missing_prompts = [p for p in config["prompts"] if p["name"] in missing_prompt_names]
    return missing_prompts


def execute_llama_run(cmd):
    try:
        output = subprocess.check_output(cmd,
            stderr=subprocess.PIPE)
        output =  output.decode("utf-8").rstrip("\u001b[0m\n")
    except subprocess.CalledProcessError as e:
        output = (f"llama-run failed with returncode {e.returncode}"
                f", stdout={e.stdout.decode('utf-8')}"
                f", stderr={e.stderr.decode('utf-8')}")
    return output


def run_prompts_using_model(llama_run_cmd, store_result_fn,
        model_name, model_path, prompts, context_len, use_cuda):
    for prompt in prompts:
        prompt_name, prompt_str = prompt["name"], prompt["prompt"]
        print(model_name, prompt_name)
        cmd = [llama_run_cmd]
        cmd.extend(["-c", str(context_len)])
        if use_cuda:
            cmd.extend(["-ngl", "1000"])
        cmd.extend([model_path, prompt_str])
        output = execute_llama_run(cmd)
        store_result_fn(model_name, prompt_name, prompt_str, output)


def store_result(results_file, results_list, model_name,
        prompt_name, prompt_str, response):
    result = {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "prompt": prompt_str,
        "response": response
    }
    results_list.append(result)
    with open(results_file, "w") as f:
        json_str = json.dumps(results_list, indent=4)
        print(json_str, file=f)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-run-cmd", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--local-model", type=str, required=False)
    parser.add_argument("--cuda", action="store_true", required=False)
    args = parser.parse_args()

    config = json.loads(open(args.config, "r").read())
    context_len = config["settings"]["context_len"]
    if os.path.isfile(args.results):
        results_list = json.loads(open(args.results, "r").read())
    else:
        results_list = []
    
    def store_result_fn(model_name, prompt_name,
            prompt_str, response):
        store_result(args.results, results_list, model_name,
            prompt_name, prompt_str, response)

    if args.local_model:
        prompts = config["prompts"]
        results_list = []
        run_prompts_using_model(args.llama_run_cmd, store_result_fn,
                "local_model", args.local_model, prompts, context_len,
                args.cuda)
    else:
        for model in config["models"]:
            prompts = get_prompts_for_model(config, results_list,
                    model["name"])
            if prompts == []:
                continue
            model_path = download_model(model["url"])
            run_prompts_using_model(args.llama_run_cmd, store_result_fn,
                    model["name"], model_path, prompts, context_len,
                    args.cuda)
            remove_downloaded_model(model["url"])


if __name__ == "__main__":
    main()

