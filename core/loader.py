import os
import hashlib
import numpy as np
from PIL import Image
from dataclasses import dataclass
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
    hist: np.ndarray = None

    orb_kp: list = None
    orb_des: np.ndarray = None


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


def _read_image_rgb(filepath: str) -> np.ndarray:
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
        return arr
    else:
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)


def _compute_phash(gray_256: np.ndarray, hash_size: int = 16):
    from PIL import Image
    import imagehash
    gray_pil = Image.fromarray(gray_256)
    ph = imagehash.phash(gray_pil, hash_size=hash_size)
    dh = imagehash.dhash(gray_pil, hash_size=hash_size)
    ah = imagehash.average_hash(gray_pil, hash_size=hash_size)
    return ph, dh, ah


def _compute_histogram(rgb_256: np.ndarray) -> np.ndarray:
    hist_r = np.histogram(rgb_256[:,:,0], bins=64, range=(0,256))[0]
    hist_g = np.histogram(rgb_256[:,:,1], bins=64, range=(0,256))[0]
    hist_b = np.histogram(rgb_256[:,:,2], bins=64, range=(0,256))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _compute_orb(gray_256: np.ndarray, max_features: int = 2000):
    import cv2
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(gray_256, None)
    return kp, des


def load_image_info(filepath: str, base_dir: str,
                    hash_size: int = 16,
                    compute_orb: bool = True,
                    orb_max_features: int = 2000) -> Optional[ImageInfo]:
    try:
        rel_path = os.path.relpath(filepath, base_dir)
        file_size = os.path.getsize(filepath)
        md5 = _compute_md5(filepath)

        rgb = _read_image_rgb(filepath)
        h, w = rgb.shape[:2]

        small = np.array(Image.fromarray(rgb).resize((256, 256)))
        gray_256 = np.dot(small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        rgb_256 = small

        phash, dhash, ahash = _compute_phash(gray_256, hash_size)
        hist = _compute_histogram(rgb_256)

        import cv2
        canny_256 = cv2.Canny(gray_256, 30, 100)

        orb_kp = None
        orb_des = None
        if compute_orb:
            try:
                orb_kp, orb_des = _compute_orb(gray_256, orb_max_features)
            except Exception:
                pass

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
            hist=hist,
            canny_256=canny_256,
            orb_kp=orb_kp,
            orb_des=orb_des,
        )
    except Exception as e:
        print(f"  [WARN] Cannot load: {filepath} - {e}")
        return None
