from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torchvision.transforms
from PIL import Image
from PIL import ImageMath
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from medpy.io import load
from skimage.transform import iradon, radon
import numpy as np
import scipy.io as io
import json

epsilon = 1e-8

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dose_of(dataroot):
    """从 dataroot 目录名推导剂量号：'train_mat_5'->'5'，'train_mat'->'1'。"""
    base = os.path.basename(os.path.normpath(dataroot))
    return base.replace('train_mat', '').strip('_') or '1'


def _preload_dir(dose):
    """预加载 memmap 所在目录（默认 ../preload，可用环境变量 PET_PRELOAD_DIR 覆盖）。"""
    default = os.path.join(_PROJECT_ROOT, '..', 'preload')
    return os.path.join(os.environ.get('PET_PRELOAD_DIR', default), str(dose))


def _load_scale_map(dataroot):
    """从 normalize_scales/{d}_scale.json 读取 per-subject scale 表。

    dataroot 形如 '.../train_mat_1'（或 '.../train_mat'），由后缀推导剂量 d。
    找不到表时返回空 dict（等价于不归一化）。
    """
    dose = _dose_of(dataroot)
    path = os.path.join(_PROJECT_ROOT, 'normalize_scales', f'{dose}_scale.json')
    if not os.path.exists(path):
        print(f'[LRHRDataset] WARNING: scale map not found: {path} '
              f'(fall back to no normalization)')
        return {}
    with open(path) as f:
        return json.load(f)


def MatrixToImage(data):
    if (data.max() > 2):
        data = (data - data.min()) / (data.max() - data.min())
    data = data * 255
    # data=np.flipud(data)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



def make_l3D(img, path, index):
    if index - 2 < 0:
        result = img
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        result = torch.Tensor(image_h)
    if index - 1 < 0:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    result = torch.cat((result, img), 0)

    if index + 1 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    if index + 2 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 0:128, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    return result

def make_h3D(img,path,index):

    if index - 2 < 0:
        result = img
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index-2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        result = torch.Tensor(image_h)
    if index - 1 < 0:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index - 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    result = torch.cat((result, img), 0)

    if index + 1 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 1) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    if index + 2 > 127:
        result = torch.cat((result, img), 0)
    else:
        image_path = '_'.join(path.split('_')[:-1]) + '_' + str(index + 2) + '.mat'
        image = io.loadmat(image_path)['img']
        image_h = image[:, 128:256, :]
        new_result = torch.Tensor(image_h)
        result = torch.cat((result, new_result), 0)

    return result


def make_psd(img):
    gen_imgs = img.permute(1, 2, 0)
    img_numpy = gen_imgs[:, :, :].cpu().detach().numpy()
    img_numpy = np.reshape(img_numpy, (128, 128))
    # img_gray = torch.Tensor(img_numpy)
    # img_gray = RGB2gray(img_numpy)
    fft = np.fft.fft2(img_numpy)
    fshift = np.fft.fftshift(fft)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    if (magnitude_spectrum.max() - magnitude_spectrum.min())!= 0:
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
        magnitude_spectrum = magnitude_spectrum[np.newaxis, :].astype(np.float32)
    else:
        magnitude_spectrum = np.zeros_like(img)
    return magnitude_spectrum

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=64, r_resolution=64, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.path = Util.get_paths_from_images(
            '{}'.format(dataroot))
        self.dataset_len = len(self.path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        # number of subjects (negative sampling indexes subjects, not slices)
        self.num = max(1, int(self.data_len) // 128)
        self.len = 0
        self.scale_map = _load_scale_map(dataroot)
        self._init_memmap(dataroot)

    def _init_memmap(self, dataroot):
        """加载预加载的 memmap（共享 OS page cache，各 worker 零拷贝）；找不到则回退 loadmat。"""
        self._mm = None
        self._subjects = None
        self._subject_idx = {}
        self._neg_scales = None
        d = _preload_dir(_dose_of(dataroot))
        data_file = os.path.join(d, 'data.bin')
        subjects_file = os.path.join(d, 'subjects.json')
        if os.path.exists(data_file) and os.path.exists(subjects_file):
            with open(subjects_file) as f:
                self._subjects = json.load(f)
            self._subject_idx = {s: i for i, s in enumerate(self._subjects)}
            self._mm = np.memmap(data_file, dtype=np.float32, mode='r',
                                 shape=(len(self._subjects), 128, 256, 128))
            self._neg_scales = [float(self.scale_map.get(s, 1.0)) for s in self._subjects]
            print(f'[LRHRDataset] memmap loaded: {len(self._subjects)} subjects <- {data_file}')
        else:
            print(f'[LRHRDataset] memmap not found ({data_file}); fallback to loadmat')

    @staticmethod
    def _context(vol, idx, half):
        """从单 subject 体积 (128,256,128) 取 5 切片上下文 -> (5,128,128)。

        half=0 -> lpet(rows 0:128)，half=1 -> hpet(rows 128:256)。
        边界行为与原 make_l3D/make_h3D 一致：越界用中心切片 idx 补齐。
        """
        rows = slice(0, 128) if half == 0 else slice(128, 256)
        out = []
        for o in (-2, -1, 0, 1, 2):
            s = idx + o if 0 <= idx + o <= 127 else idx
            out.append(vol[s, rows])
        return torch.from_numpy(np.stack(out)).float()

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.len == 128:
            self.len = 0
        slice_idx = self.len
        self.len += 1

        subject_id = os.path.basename(self.path[index]).rsplit('_', 1)[0]
        # ---- per-subject 归一化（口径与推理 load_subject_volume 共用 normalize_common）----
        scale = float(self.scale_map.get(subject_id, 1.0))

        if self._mm is not None:
            si = self._subject_idx[subject_id]
            full = self._mm[si, slice_idx]                              # (256,128)
            img_spet = torch.tensor(full[0:128]).unsqueeze(0)   # (1,128,128)，torch.tensor 复制出可写张量
            img_hpet = torch.tensor(full[128:256]).unsqueeze(0) # (1,128,128)
            img_3d_l = self._context(self._mm[si], slice_idx, half=0)
            img_3d_h = self._context(self._mm[si], slice_idx, half=1)
        else:
            image_path = '_'.join(os.path.join(self.path[index]).split('_')[:-1]) + '_' + str(slice_idx) + '.mat'
            image = io.loadmat(image_path)['img']
            img_spet = torch.Tensor(image[:, 0:128, :])
            img_hpet = torch.Tensor(image[:, 128:256, :])
            img_3d_l = make_l3D(img_spet, self.path[index], slice_idx)
            img_3d_h = make_h3D(img_hpet, self.path[index], slice_idx)

        if self.need_LR:
            img_lpet = img_spet

        img_Lpsd = make_psd(img_spet)      # 频谱自归一（log+minmax），尺度不变，无需处理
        img_Hpsd = make_psd(img_hpet)

        # 统一除以本 subject 的 scale（L3D/H3D 是同一 subject 的 5 切片，共用同一 scale）
        img_spet = img_spet / scale
        img_hpet = img_hpet / scale
        img_3d_l = img_3d_l / scale
        img_3d_h = img_3d_h / scale
        if self.need_LR:
            img_lpet = img_lpet / scale

        # ---- 负样本（10 个，来自随机其它 subject，用其自身 scale）----
        for i in range(10):
            negative_index = random.randint(0, self.num - 1)
            while negative_index == int(index/128):
                negative_index = random.randint(0, self.num - 1)
            j = int(np.random.normal(0, 2, 1)[0])
            t = self.len + j - 1
            if not (0 <= t <= 127):
                t = self.len - 1
            if self._mm is not None:
                neg_scale = self._neg_scales[negative_index]
                neg_img = torch.tensor(self._mm[negative_index, t, 128:256]).unsqueeze(0) / neg_scale
            else:
                negative_path = os.path.join(self.path[negative_index*128])
                neg_subject_id = os.path.basename(negative_path).rsplit('_', 1)[0]
                neg_scale = float(self.scale_map.get(neg_subject_id, 1.0))
                negative_image_path = '_'.join(negative_path.split('_')[:-1]) + '_' + str(t) + '.mat'
                neg_img = torch.Tensor(io.loadmat(negative_image_path)['img'][:, 128:256, :]) / neg_scale
            if i == 0:
                negative_hpet = neg_img
            else:
                negative_hpet = torch.cat((negative_hpet, neg_img), 0)

        if self.need_LR:
            return {'LR': img_lpet, 'HR': img_hpet, 'SR': img_spet, 'NHR': negative_hpet,
                    'LP': img_Lpsd, 'HP': img_Hpsd, 'L3D': img_3d_l, 'H3D': img_3d_h,
                    'Index': index}
        else:
            return {'HR': img_hpet, 'SR': img_spet, 'LP': img_Lpsd, 'HP': img_Hpsd, 'NHR': negative_hpet,
                    'L3D': img_3d_l, 'H3D': img_3d_h, 'Index': index}


if __name__ == '__main__':

    from data.LRHR_dataset import LRHRDataset as D

    dataset = D(
        dataroot='E:\数据集\Desktop\\train_mat',
        datatype='jpg',
        l_resolution=64,
        r_resolution=64,
        split='train',
        data_len=-1,
        need_LR=False
    )
    dataset.__getitem__(3)

