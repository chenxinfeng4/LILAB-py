import numpy as np
from subprocess import Popen, PIPE
import re
from multiprocessing import Value
import ffmpeg

# %%
class ffmpeg_writer:
    igpu = 0

    def __init__(self):
        pass

    def __del__(self):
        if hasattr(self, 'process'):
            self.release()

    @staticmethod
    def _get_num_gpu():
        p = Popen(['nvidia-smi'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"")

        pattern = re.compile(r'\d+  NVIDIA')
        nv_info = pattern.findall(output.decode())
        num_gpu = len(nv_info)
        return num_gpu

    @staticmethod
    def VideoWriter(filename, fourcc, fps, frameSize, gpu=None):
        # frameSize = (width, height)
        if gpu is None:
            ffmpeg_writer.igpu += 1
            gpu = ffmpeg_writer.igpu % ffmpeg_writer._get_num_gpu()
        else:
            gpu = int(gpu) % ffmpeg_writer._get_num_gpu()

        vcodec = 'hevc_nvenc'
        width, height = frameSize
        vid = ffmpeg_writer()
        vid.fps, vid.width, vid.height = fps, width, height
        vid.process = (
            ffmpeg
                .input('pipe:',  r=fps, format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
                .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=fps, gpu=gpu)
                .global_args("-loglevel", "warning")
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        return vid

    def write(self, img):
        img = img.astype(np.uint8).tobytes()
        self.process.stdin.write(img)

    def release(self):
        self.process.stdin.close()
        self.process.wait()
