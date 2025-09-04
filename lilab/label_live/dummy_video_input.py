import ffmpegcv
from ffmpegcv import FFmpegReader
import tqdm

class FFmpegReaderNV_list(FFmpegReader):
    @staticmethod
    def VideoReader(
        filename_list,
        *args,
        **kwargs
    ):
        assert isinstance(filename_list, list)
        vid = FFmpegReaderNV_list()
        vid_l = [ffmpegcv.VideoCaptureNV(filename, *args, **kwargs)
                 for filename in filename_list]
        vid.filename = 'concatall.mp4'
        vid.size, vid.width, vid.height = vid_l[0].size, vid_l[0].width, vid_l[0].height
        vid.crop_width, vid.crop_height = vid_l[0].crop_width, vid_l[0].crop_height
        vid.origin_width, vid.origin_height = vid_l[0].origin_width, vid_l[0].origin_height
        vid.out_numpy_shape = vid_l[0].out_numpy_shape
        vid.fps, vid.codec = vid_l[0].fps, vid_l[0].codec
        vid.count = sum([vid_l[i].count for i in range(len(vid_l))])
        vid.count_l = [vid_l[i].count for i in range(len(vid_l))]
        vid.vid_l = vid_l
        vid.vid_i = 0
        return vid

    def read(self):
        ret, frame = self.vid_l[self.vid_i].read()
        if not ret:
            self.vid_i += 1
            if self.vid_i >= len(self.vid_l):
                return False, None
            ret, frame = self.vid_l[self.vid_i].read()
        return ret, frame
    

files="""
2024-12-30_17-29-50_l1_sm1pm4
2024-12-30_17-51-03_l1_sm2pm5
2024-12-30_18-34-58_l1_sm4pm1
2024-12-30_19-00-29_l1_sm5pm2
2024-12-31_15-04-32_l2_sm1pm5
2024-12-31_15-32-22_l2_sm2pm6
2024-12-31_16-17-22_l2_sm4pm2
2024-12-31_16-41-13_l2_sm5pm3
2025-01-01_14-30-03_l3_sm1pm6
2025-01-01_14-53-02_l3_sm2pm1
2025-01-01_15-42-22_l3_sm4pm3
2025-01-01_16-08-30_l3_sm5pm4
2025-01-02_15-50-02_l4_sm1_pm1
2025-01-02_16-26-11_l4_sm2_pm2
2025-01-02_17-16-17_l4_sm4_pm4
2025-01-02_17-41-47_l4_sm5_pm5
2025-01-03_15-01-49_l5_sm1_pm2
2025-01-03_15-27-41_l5_sm2_pm3
2025-01-03_16-17-40_l5_sm4_pm5
2025-01-03_16-44-57_l5_sm5_pm6
2025-01-04_15-00-24_l6_sm1_pm3
2025-01-04_15-24-28_l6_sm2_pm4
2025-01-04_16-13-43_l6_sm4_pm6
2025-01-04_16-42-30_l6_sm5_pm1
2025-01-05_15-04-37_l7_sm1_pm6
2025-01-05_15-35-26_l7_sm2_pm5
2025-01-05_16-27-40_l7_sm4_pm1
2025-01-05_16-54-48_l7_sm5_pm2
2025-01-06_14-54-36_l8_sm1_pm5
2025-01-06_15-21-01_l8_sm2_pm6
2025-01-06_16-12-57_l8_sm4_pm2
2025-01-06_16-39-33_l8_sm5_pm3
"""

files = [f'/DATA/zhongzhenchao/2501chr2-shank3/training/{v}.mp4' for v in files.split('\n') if v]


# 867647
def get_vid_dummy_inputs():
    return FFmpegReaderNV_list.VideoReader(files, pix_fmt='gray')
