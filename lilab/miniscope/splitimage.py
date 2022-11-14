import os
import argparse
import sys
from itertools import count

def main(step):
    flag_end = '\xD9FF'
    file_not_end = True
    chunk_size = 1024
    for i in count():
        do_write = i%step == 0

        while True:
            # read stdin  until flag_end1
            c = sys.stdin.read(chunk_size)
            idx = c.find(flag_end)
            if idx != -1:
                img = c[:idx+2]
        sys.stdout.write(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert video to 5fps')
    parser.add_argument('-s', '--step', type=int, default=5, help='step size')
    args = parser.parse_args()
    main(args.step)
