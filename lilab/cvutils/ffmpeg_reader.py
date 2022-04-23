import numpy as np
from subprocess import Popen, PIPE
import subprocess
import re
from ffmpegcv.video_info import get_info, decoder_to_nvidia

def run_async(args):
    quiet = True
    stderr_stream = subprocess.DEVNULL if quiet else None
    return subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_stream, shell=True
    )

# %%
class ffmpeg_reader:
    igpu = 0

    def __init__(self):
        self._fakewh = 10
        pass

    def __del__(self):
        if hasattr(self, 'process'):
            pass

    def __len__(self):
        return self.count

    def __iter__(self):
        return self

    def __next__(self):
        ret, img = self.read()
        if ret:
            return img
        else:
            raise StopIteration

    @staticmethod
    def _get_num_gpu():
        p = Popen(['nvidia-smi'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"")

        pattern = re.compile(r'\d+  NVIDIA')
        nv_info = pattern.findall(output.decode())
        num_gpu = len(nv_info)
        return num_gpu

    @staticmethod
    def VideoReader(filename, crop_xywh=None, gpu=None, pix_fmt='bgr24'):
        if gpu is None:
            ffmpeg_reader.igpu += 1
            gpu = ffmpeg_reader.igpu % ffmpeg_reader._get_num_gpu()
        else:
            gpu = int(gpu) % ffmpeg_reader._get_num_gpu()

        assert pix_fmt in ['rgb24', 'bgr24']

        vid = ffmpeg_reader()
        videoinfo = get_info(filename)
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.codec = videoinfo.codec
        codec = decoder_to_nvidia(vid.codec)

        if crop_xywh:
            x, y, w, h = crop_xywh
            top, bottom, left, right = y, vid.height - (y + h), x, vid.width - (x + w)  #crop length
            vid.width, vid.height = w, h
            cropopt = f'-crop {top}x{bottom}x{left}x{right}'
        else:
            cropopt = ''
        args = (f'ffmpeg -hwaccel cuda -hwaccel_device {gpu} '
                f'-vcodec {codec} {cropopt} -r {fps} -i "{filename}" '
                f'-pix_fmt {pix_fmt} -r {fps} -f rawvideo pipe:')
        print(args)
        vid.process = run_async(args)
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3)
        if not in_bytes:
            return False, None
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        return True, img

    def release(self):
        self.process.terminate()
        self.process.wait()


class ffmpeg_resize_reader(ffmpeg_reader):
    @staticmethod
    def VideoReader(filename, dst_size, crop_xywh=None, gpu=None, pix_fmt='bgr24'):
        if gpu is None:
            ffmpeg_reader.igpu += 1
            gpu = ffmpeg_reader.igpu % ffmpeg_reader._get_num_gpu()
        else:
            gpu = int(gpu) % ffmpeg_reader._get_num_gpu()

        assert pix_fmt in ['rgb24', 'bgr24']

        vid = ffmpeg_reader()
        videoinfo = get_info(filename)
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = fps = videoinfo.fps
        vid.count = videoinfo.count
        codec = decoder_to_nvidia(vid.codec)

        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
        else:
            crop_w, crop_h = vid.width, vid.height
        if not (isinstance(dst_size, tuple) or isinstance(dst_size, list)):
            dst_size = [dst_size, dst_size]
        dst_width, dst_height = dst_size
        re_width, re_height = crop_w/(crop_h / dst_height) , dst_height
        if re_width > dst_width:
            re_width, re_height = dst_width, crop_h/(crop_w / dst_width)
        re_width, re_height = int(re_width), int(re_height)
        scaleopt = f'-vf scale_cuda={re_width}:{re_height},hwdownload,format=nv12'
        xpading, ypading = (dst_width - re_width) // 2, (dst_height - re_height) // 2
        padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
        filteropt = f'{scaleopt},{padopt}'
        if crop_xywh:
            x, y, w, h = crop_xywh
            top, bottom, left, right = y, vid.height - (y + h), x, vid.width - (x + w)  #crop length
            vid.width, vid.height = w, h
            cropopt = f'-crop {top}x{bottom}x{left}x{right}'
        else:
            cropopt = ''
        args = (f'ffmpeg -hwaccel cuda -hwaccel_device {gpu} -hwaccel_output_format cuda '
                f' -vcodec {codec} {cropopt} -r {fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {fps} -f rawvideo pipe:')
        print(args)
        vid.process = run_async(args)

        vid.origin_width, vid.origin_height = vid.width, vid.height
        vid.width, vid.height = dst_width, dst_height

        return vid
