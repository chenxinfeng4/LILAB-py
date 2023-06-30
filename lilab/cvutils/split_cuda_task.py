import os
import os.path as osp
import sys
import glob
import shutil
# get the cuda devices
cudas = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
cudas = cudas
ncuda = len(cudas)

filetypes = ['avi', 'mp4']
dowrytypes = 'pkl'

def split_dataset(videofolder):
    files = sum([glob.glob(osp.join(videofolder, '*.' + filetype)) for filetype in filetypes], [])
    files_chunks=[files[i:i + ncuda] for i in range(0, len(files), ncuda)]
    files_proxyfolders = [osp.join(videofolder, '.proxy' + cuda) for cuda in cudas]
    for proxyfolder in files_proxyfolders:
        shutil.rmtree(proxyfolder, ignore_errors=True)
        os.makedirs(proxyfolder)

    # hard link the files_chunks to the proxy folder
    for files_chunk, proxyfolder in zip(files_chunks, files_proxyfolders):
        for file in files_chunk:
            os.link(file, proxyfolder+'/'+osp.basename(file))
            dowryfile = osp.splitext(file)[0] + '.' + dowrytypes
            if osp.isfile(dowryfile):
                os.link(dowryfile, proxyfolder+'/'+osp.basename(dowryfile))

    return files_proxyfolders


if __name__ == '__main__':
    # get the argv
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if os.path.isdir(arg):
            videofolder = arg
            iarg = i
            break
    else:
        raise ValueError("the video folder is not specified")

    video_proxyfolders = split_dataset(videofolder)
    for i, video_proxyfolder in enumerate(video_proxyfolders):
        args[iarg] = video_proxyfolder
        argsstr = ' '.join(args) + f' --device cuda:{i}'
        print(argsstr)
