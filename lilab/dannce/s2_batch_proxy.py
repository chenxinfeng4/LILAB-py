# %%
import os
import os.path as osp
import sys
import re
from concurrent.futures import ThreadPoolExecutor
import ffmpegcv
from queue import Queue
import subprocess
import argparse

# %%
# get the stdin input string
num_gpus = ffmpegcv.video_info.get_num_NVIDIA_GPUs()

# remove the gpu-id

# %%
# get the number of gpus by torch


def run_command(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    while True:
        output = process.stdout.readline()
        if output == b"" and process.poll() is not None:
            break
        if output:
            sys.stdout.write(output)
    return process.poll()


def task(command: str, q: Queue):
    gpu = q.get()
    # use the systemcmd to run the command, and monitor the output to get the progress
    systemcmd = command + " --gpu-id " + str(gpu)  # run the command on the gpu
    os.system(systemcmd)
    # run_command(systemcmd)
    q.put(gpu)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--num-gpus", type=int, default=num_gpus, help="the number of gpus"
    )
    inputstr = sys.stdin.read()
    input_list = inputstr.split("\n")
    print("inputstr", input_list)
    pattern = re.compile(r"(--gpu-id \d+)")
    input_clean_list = [pattern.sub("", i) for i in input_list]
    print("input_clean_list", input_clean_list)

    pool = ThreadPoolExecutor(max_workers=4)
    gpu_queue = Queue(maxsize=num_gpus)
    for i in range(num_gpus):
        gpu_queue.put(i)

    for task_name in input_clean_list:
        pool.submit(task, task_name, gpu_queue)
