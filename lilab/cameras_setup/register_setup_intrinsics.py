#python lilab.cameras_setup.regitster_setup_intrinsics xx.json bob2
import argparse
import shutil
import os
import os.path as osp


def main(file:str, new_setup_name:str):
    current_dir = osp.dirname(osp.abspath(__file__))
    assert osp.exists(file), file
    os.makedirs(osp.join(current_dir, new_setup_name), exist_ok=True)
    out_file = osp.join(current_dir, new_setup_name, 'intrinsics_calib.json')
    if osp.exists(out_file):
        print(f'{out_file} already exists. Will be overwrited')
    shutil.copy(file, out_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('intrin_json', type=str, help='intrinsics file')
    parser.add_argument('new_setup_name', type=str, help='new setup name')
    args = parser.parse_args()
    main(args.intrin_json, args.new_setup_name)
