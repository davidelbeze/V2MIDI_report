import argparse
import os
import json
import time
import torch
import torch.multiprocessing as mp
from math import ceil
import requests
import shutil
from queue import Queue
from threading import Thread


def get_config(args):
    number_of_nodes = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    node_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))

    entries_by_node = ceil(args.count / number_of_nodes)
    print(f"Node configs entries by node: {entries_by_node}")
    print(f"Node configs node ID: {node_id}")
    node_configs = []
    for i in range(entries_by_node):
        config_index = node_id * entries_by_node + i
        if config_index < args.count:
            node_configs.append(f"{args.dataset_path}/midi_parseq_{config_index + 1}/parseq_{config_index + 1}.json")

    print(f"Node configs length: {len(node_configs)}")
    print(f"Node configs start: {node_configs[0]}")
    print(f"Node configs end: {node_configs[-1]}")

    return node_configs


def get_device_config():
    if torch.cuda.is_available():
        local_world_size = torch.cuda.device_count()
        backend, device = 'nccl', 'cuda'
    elif torch.backends.mps.is_available():
        local_world_size = 4
        backend, device = 'gloo', 'mps'
    else:
        local_world_size = 4
        backend, device = 'gloo', 'cpu'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    return local_world_size, backend, device


def launch_a11_backend(rank, local_world_size, backend, device, args, ports):
    port = ports[rank]
    os.system("module load pytorch-gpu/py3/2.1.1")
    os.system("module load git")
    os.environ["PYTHONUSERBASE"] = "/gpfswork/rech/fkc/uhx75if/.local_automatic1111"
    os.environ["PYTHONPATH"] = "/gpfswork/rech/fkc/uhx75if/video2midi/stable-diffusion-webui"
    os.environ["SD_WEBUI_LOG_LEVEL"] = "CRITICAL"
    os.environ['PYTHON_LOG_LEVEL'] = 'CRITICAL'
    commande = f"python /gpfswork/rech/fkc/uhx75if/video2midi/stable-diffusion-webui/launch.py --api --port {port} --device-id {str(rank)} --opt-sdp-attention --disable-model-loading-ram-optimization --deforum-api --disable-console-progressbars"
    os.system(commande)


def post_request(config, ports, gpu_id, args):
    url = f"http://127.0.0.1:{ports[gpu_id]}"
    settings_path = args.settings_path

    with open(settings_path, 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    with open(config, 'r') as parseq_file:
        parseq_data = json.load(parseq_file)

    deforum_settings['parseq_manifest'] = json.dumps(parseq_data)
    deforum_settings["parseq_non_schedule_overrides"] = True

    # Extract the index from the config file path and set the batch_name
    config_index = config.split('_')[-1].split('.')[0]
    deforum_settings["batch_name"] = f"batch_{config_index}"

    # is file is already processed, skip
    gen_path = f"/gpfsscratch/rech/fkc/uhx75if/midi_videos_output/img2img-images/batch_{config_index}/"
    if os.path.exists(gen_path):
        if any(".mp4" in file for file in os.listdir(gen_path)):
            print(f"File already processed: {config}")
            return 'already'

    payload = {
        "deforum_settings": [deforum_settings],
        "options_overrides": {
            "deforum_save_gen_info_as_srt": True,
            "deforum_save_gen_info_as_srt_params": {}
        }
    }

    response = requests.post(url=f'{url}/deforum_api/batches', json=payload, timeout=(1, 2))
    response.raise_for_status()
    job_ids = response.json().get("job_ids", [])

    print(f"Config index: {config_index}")
    print(f"Job IDs: {job_ids}")

    def wait_for_job_to_complete(job_id):
        status_url = f"{url}/deforum_api/jobs/{job_id}"
        while True:
            response = requests.get(status_url)
            response.raise_for_status()
            status = response.json()
            if status["status"] in ["SUCCEEDED", "FAILED"]:
                return status
            time.sleep(5)

    for job_id in job_ids:
        job_status = wait_for_job_to_complete(job_id)
        if job_status["status"] == "SUCCEEDED":
            print(f"Job {job_id} succeeded")
            output_dir = job_status["outdir"]
            print(f"Output directory: {output_dir}")

            # Copy the associated MIDI file to the output directory
            midi_index = int(config.split('_')[-1].split('.')[0])
            midi_path = f"{args.dataset_path}/midi_parseq_{midi_index}/midi_{midi_index}.mid"
            target_midi_path = os.path.join(output_dir, f"midi_{midi_index}.mid")
            shutil.copy(midi_path, target_midi_path)

            print(f"MIDI file saved to: {target_midi_path}")
        else:
            print(f"Job {job_id} failed with status: {job_status}")

    return 'done'


def worker(gpu_id, ports, task_queue, args):
    launched = False
    out = "already"

    while out == "already":
        try:
            config = task_queue.get()
            out = post_request(config, ports, gpu_id, args)

            if out == 'already' or out == 'done':
                task_queue.task_done()

        except Exception as e:
            while not launched:
                try:
                    out = post_request(config, ports, gpu_id, args)
                    if out == 'done':
                        launched = True
                    task_queue.task_done()
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)

    config = task_queue.get()
    while config is not None:
        post_request(config, ports, gpu_id, args)
        task_queue.task_done()
        config = task_queue.get()


def main(args):
    node_configs = get_config(args)
    local_world_size, backend, device = get_device_config()
    ports = [52361 + i for i in range(local_world_size)]

    context = mp.spawn(
        fn=launch_a11_backend,
        args=(local_world_size, backend, device, args, ports),
        nprocs=local_world_size,
        join=False
    )

    task_queue = Queue(maxsize=10)

    workers = []
    for gpu_id in range(local_world_size):
        thread = Thread(target=worker, args=(gpu_id, ports, task_queue, args))
        thread.start()
        workers.append(thread)

    for config in node_configs:
        task_queue.put(config)

    task_queue.join()

    for _ in range(local_world_size):
        task_queue.put(None)

    for thread in workers:
        thread.join()


if __name__ == "__main__":

    # set python log level to error

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run the script locally")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--settings_path", type=str, required=True, help="Path to the Deforum settings file")
    parser.add_argument("--path", type=str, help="Path to store the outputs")
    parser.add_argument("--count", type=int, default=40000, help="Total number of configurations")
    parsed_args = parser.parse_args()
    main(parsed_args)
