import os
import hashlib
import threading
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional


IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
    '.gif', '.webp',
}


@dataclass
class ImageInfo:
    path: str
    rel_path: str
    size: tuple
    file_size: int
    md5: str
    group_info: dict

    rgb_256: np.ndarray = None
    gray_256: np.ndarray = None
    canny_256: np.ndarray = None
    phash: object = None
    dhash: object = None
    ahash: object = None
    whash: object = None
    hist: np.ndarray = None

    orb_kp: list = None
    orb_des: np.ndarray = None

    sift_kp: list = None
    sift_des: np.ndarray = None
    sift_scale: float = 1.0
    sift_loaded: bool = False

    bow_keys: set = None

    sift_lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure_sift(self, max_dim: int = 1024, nfeatures: int = 2000) -> bool:
        """懒加载高分辨率 SIFT 特征 (关键点坐标已换算回原图像素)。

        只对进入候选对的图片调用, 避免全量计算的浪费。
        线程安全: 锁在计算完成前不会释放, 调用方返回时特征必然就绪。
        返回 True 表示特征可用。
        """
        if self.sift_des is not None:
            return True
        with self.sift_lock:
            if self.sift_des is not None:
                return True
            try:
                arr = read_gray_capped(self.path, max_dim=max_dim)
                h, w = arr.shape[:2]
                ow, oh = self.size
                scale_x = ow / w
                scale_y = oh / h
                sift = cv2.SIFT_create(nfeatures=nfeatures,
                                       contrastThreshold=0.03,
                                       edgeThreshold=12)
                kp, des = sift.detectAndCompute(arr, None)
                if des is not None and len(kp) > 0:
                    kp = [cv2.KeyPoint(x=p.pt[0] * scale_x, y=p.pt[1] * scale_y,
                                       size=p.size, angle=p.angle,
                                       response=p.response, octave=p.octave,
                                       class_id=p.class_id) for p in kp]
                    self.sift_kp = kp
                    self.sift_des = des
                    self.sift_scale = 1.0
                    self.sift_loaded = True
            except Exception as e:
                print(f"  [WARN] ensure_sift 失败: {self.path}: {e}")
                self.sift_kp = None
                self.sift_des = None
            return self.sift_des is not None


def scan_images(directory: str, extensions: set = None) -> list[str]:
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    directory = os.path.abspath(directory)
    files = []
    for root, dirs, filenames in os.walk(directory):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, fn))
    return sorted(files)


def _compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_image_rgb(filepath: str, max_dim: int = 4096) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            arr = page.asarray()
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if max_dim and max(arr.shape[:2]) > max_dim:
            scale = max_dim / max(arr.shape[:2])
            arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        return arr
    else:
        img = Image.open(filepath)
        if max_dim and max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)


def read_gray_capped(filepath: str, max_dim: int = 1024) -> np.ndarray:
    """读取灰度图, 长边限制在 max_dim 以内。"""
    rgb = _read_image_rgb(filepath, max_dim=max_dim)
    if rgb.ndim == 2:
        return rgb
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def _compute_phash(gray_256: np.ndarray, hash_size: int = 16):
    from PIL import Image
    import imagehash
    gray_pil = Image.fromarray(gray_256)
    ph = imagehash.phash(gray_pil, hash_size=hash_size)
    dh = imagehash.dhash(gray_pil, hash_size=hash_size)
    ah = imagehash.average_hash(gray_pil, hash_size=hash_size)
    wh = None
    try:
        wh = imagehash.whash(gray_pil, hash_size=hash_size)
    except Exception:
        pass
    return ph, dh, ah, wh


def _compute_histogram(rgb_256: np.ndarray) -> np.ndarray:
    hist_r = np.histogram(rgb_256[:,:,0], bins=64, range=(0,256))[0]
    hist_g = np.histogram(rgb_256[:,:,1], bins=64, range=(0,256))[0]
    hist_b = np.histogram(rgb_256[:,:,2], bins=64, range=(0,256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _compute_bow_keys(rgb: np.ndarray, max_dim: int = 512, nfeatures: int = 1000) -> set:
    """从已解码的 rgb 图提取 ORB 描述子键集合 (无需额外 IO)。"""
    try:
        h, w = rgb.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            small = cv2.resize(rgb, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        gray = small if small.ndim == 2 else np.dot(
            small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        orb = cv2.ORB_create(nfeatures=nfeatures)
        _, des = orb.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return set()
        view = des.view(np.uint32).reshape(-1, 8)
        return set(int(k) for k in view[:, 0])
    except Exception:
        return set()


def load_image_info(filepath: str, base_dir: str,
                    hash_size: int = 16) -> Optional[ImageInfo]:
    try:
        rel_path = os.path.relpath(filepath, base_dir)
        file_size = os.path.getsize(filepath)
        md5 = _compute_md5(filepath)

        rgb = _read_image_rgb(filepath)
        h, w = rgb.shape[:2]

        small = np.array(Image.fromarray(rgb).resize((256, 256)))
        gray_256 = np.dot(small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        rgb_256 = small

        phash, dhash, ahash, whash = _compute_phash(gray_256, hash_size)
        hist = _compute_histogram(rgb_256)

        canny_256 = cv2.Canny(gray_256, 30, 100)

        return ImageInfo(
            path=filepath,
            rel_path=rel_path,
            size=(w, h),
            file_size=file_size,
            md5=md5,
            group_info={},
            rgb_256=rgb_256,
            gray_256=gray_256,
            phash=phash,
            dhash=dhash,
            ahash=ahash,
            whash=whash,
            hist=hist,
            canny_256=canny_256,
            sift_kp=None,
            sift_des=None,
            bow_keys=_compute_bow_keys(rgb),
        )
    except Exception as e:
        print(f"  [WARN] Cannot load: {filepath} - {e}")
        return None
