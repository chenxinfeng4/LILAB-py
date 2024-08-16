# python -m lilab.cvutils_new.get_file_ctime xxx/xxx/
import os
import sys
import glob
import datetime
import argparse


def get_file_ctime(file_path):
    return datetime.datetime.fromtimestamp(os.path.getctime(file_path))


def main(file_path):
    files_ctime = [get_file_ctime(filename) for filename in file_path]
    outfile = os.path.join(os.path.dirname(file_path[0]), "file_ctime.txt")
    with open(outfile, "w") as f:
        for filename, ctime in zip(file_path, files_ctime):
            f.write("{}\t{}\n".format(filename, ctime))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get file creation time")
    parser.add_argument(
        "file_path", type=str, default=None, help="path to file or folder"
    )

    args = parser.parse_args()
    file_path = args.file_path
    assert os.path.exists(file_path), "file_path not exists"
    if os.path.isfile(file_path):
        file_path = [file_path]
    elif os.path.isdir(file_path):
        file_path = glob.glob(os.path.join(file_path, "*.mp4"))
        assert len(file_path) > 0, "no video found"
    else:
        raise ValueError("image_path is not a file or folder")

    main(file_path)
