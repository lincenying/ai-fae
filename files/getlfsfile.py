import argparse
import os
import sys
from openmind_hub import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_to_search",
        type=str,
        help="Path to model files",
        default=None,
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Repo id",
        default=None,
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Repo id",
        default=None,
    )
    args = parser.parse_args()
    return args


args = parse_args()
directory_to_search = "./" + args.directory_to_search
repo_id = args.repo_id

token = args.token

uploaded_file_log = directory_to_search + "/uploaded_files.log"


def mark_as_uploaded(file_name):
    with open(uploaded_file_log, "a") as log_file:
        log_file.write(file_name + "\n")


def check_if_uploaded(file_name):
    if os.path.exists(uploaded_file_log):
        with open(uploaded_file_log, "r") as log_file:
            for line in log_file:
                if file_name == line.strip():
                    return True
    return False


def replace_directory_in_path(file_path, directory_to_remove):
    if file_path.startswith(directory_to_remove):
        return file_path[len(directory_to_remove) :]
    else:
        return file_path


def get_bin_and_safetensors_files(directory):
    filenames = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in (".bin", ".safetensors", ".gguf", ".h5", ".msgpack", ".ot", ".pt", ".pth")):
                filepath = os.path.join(root, file)
                if not check_if_uploaded(filepath):
                    filenames.append(filepath)
                else:
                    print(f"{file} has already been uploaded. Skipping...")

    return filenames


if args.directory_to_search and args.repo_id:

    file_list = get_bin_and_safetensors_files(directory_to_search)

    print("Total files: " + str(len(file_list)))
    print(file_list)

    index = 1
    for upfile in file_list:
        print(
            "Start Uploading: "
            + upfile
            + "  ===>  "
            + replace_directory_in_path(upfile, directory_to_search + "/")
            + " ---- "
            + str(index)
            + "/"
            + str(len(file_list))
        )

        upload_file(token=token, path_or_fileobj=upfile, repo_id=repo_id, path_in_repo=replace_directory_in_path(upfile, directory_to_search + "/"))
        mark_as_uploaded(upfile)
        index += 1

else:
    print("directory_to_search参数和repo_id参数不能为空!")
