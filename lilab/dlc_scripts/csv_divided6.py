"""
python -m lilab.dlc_scripts.csv_divided6 --file \
    /home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/labeled-data/merged_sideframes/CollectedData_edf.csv

将一个手动标注的csv文件分割成6个csv文件，每个csv文件包含一个view
"""
import argparse

num_views = 6

def a1_load_dlc_csv(file):
    lineheaders = []
    with open(file, 'r') as f:
        for line in f:
            lineheaders.append(line)
            if line.startswith('coords'):
                break             
        else:
            raise 'Not Valid deeplabcut csv'
        body = f.readlines()
    # assert len(lineheaders) is the integer multiples of num_views
    assert len(body) % num_views == 0 and len(body) > 0
    
    # write files
    out_files = [file.replace('.csv', '_view{}.csv'.format(i+1)) for i in range(num_views)]
    for i, out_file in enumerate(out_files):
        with open(out_file, 'w') as f:
            f.writelines(lineheaders)
            f.writelines(body[i::6])

# __main__
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a deeplabcut csv file into multiple files')
    parser.add_argument('--file', type=str, required=True, help='deeplabcut csv file')
    args = parser.parse_args()
    a1_load_dlc_csv(args.file)
    