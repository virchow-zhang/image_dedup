"""AI 嵌入向量层 (可选, --ai)。

用 MobileNetV3-Small (ONNX, 约 7MB) 提取全局视觉特征,
余弦相似度召回经过重度后期处理(对比度曲线/叠加/重压缩)的近重复图片,
弥补感知哈希在强变换下的退化。

- 模型首次使用自动下载到 models/ 目录, 支持多个镜像源
- 下载失败时优雅降级: 打印警告并跳过 AI 层, 不影响其他检测
"""
import os
import sys
import urllib.request

import numpy as np

MODEL_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _download(url: str, dst: str) -> bool:
    """下载到 dst, 支持断点续传与自动重试。优先用 curl.exe (Windows 自带)。"""
    import shutil
    import subprocess

    curl = shutil.which("curl")
    try:
        if curl:
            cmd = [curl, "-L", "-C", "-", "--retry", "10", "--retry-delay", "2",
                   "--retry-all-errors", "-sS", "--connect-timeout", "30",
                   "-o", dst, url]
            r = subprocess.run(cmd, capture_output=True, timeout=900)
            if r.returncode == 0:
                return True
        req = urllib.request.Request(url, headers={"User-Agent": "image_dedup/3.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(dst, 'wb') as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"  [AI] 下载失败({url}): {e}")
        return False


def ensure_model(model_path: str) -> str:
    """确保 ONNX 模型存在, 不存在则下载。返回模型路径或 None。"""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 500_000:
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    part = model_path + ".part"
    for url in MODEL_URLS:
        print(f"  [AI] 下载嵌入模型: {url}")
        if _download(url, part):
            if os.path.getsize(part) > 500_000:
                os.replace(part, model_path)
                print(f"  [AI] 模型已保存: {model_path}")
                return model_path
    if os.path.exists(part):
        os.remove(part)
    return None


def prepare_embedding_model(model_path: str) -> str:
    """给分类模型补一个中间特征输出节点, 返回新的模型路径。

    分类模型通常只导出 logits 输出 (如 1000 维), 嵌入向量需要
    logits 之前一层的特征 (如 MobileNetV2 的 1280 维 flatten)。
    通过追加 Identity 节点将特征张量暴露为名为 "embedding" 的输出。
    """
    try:
        import onnx
        from onnx import helper
    except ImportError:
        return None

    out_path = model_path + ".embed.onnx"
    if os.path.exists(out_path):
        return out_path
    try:
        m = onnx.load(model_path)
        producer = {}
        for node in m.graph.node:
            for o in node.output:
                producer[o] = node
        logits_name = m.graph.output[0].name
        tail = producer.get(logits_name)
        if tail is None:
            return None
        # 分类头 (Reshape/Softmax) 的输入是 logits 4D 张量
        logits4d = tail.input[0]
        head = producer.get(logits4d)
        if head is None:
            return None
        # logits 头 (Conv/Gemm) 的输入即池化特征 (如 1280 维)
        emb = head.input[0]
        m.graph.node.append(helper.make_node(
            "Identity", inputs=[emb], outputs=["embedding"]))
        m.graph.output.append(helper.make_tensor_value_info(
            "embedding", onnx.TensorProto.FLOAT, None))
        onnx.save(m, out_path)
        return out_path
    except Exception:
        return None


class EmbeddingModel:
    """MobileNetV2 嵌入模型封装 (onnxruntime, CPU)。"""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        session_path = model_path
        self.session = ort.InferenceSession(
            session_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        try:
            self.output_name, self.output_dim = self._pick_output()
        except RuntimeError:
            # 分类模型没有特征输出 -> 自动补层后重载
            emb_path = prepare_embedding_model(model_path)
            if emb_path is None:
                raise
            session_path = emb_path
            self.session = ort.InferenceSession(
                session_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name, self.output_dim = self._pick_output()
        self.batch_size = 32

    def _pick_output(self):
        """从模型中挑出池化特征输出 (排除 1000 类 logits)。

        兼容 2D [1, D] 与 4D [1, D, 1, 1] 两种特征张量。
        """
        best = None
        for o in self.session.get_outputs():
            s = o.shape
            d = None
            if len(s) == 2 and s[1] is not None:
                d = s[1]
            elif len(s) == 4 and s[0] == 1 and s[1] is not None:
                d = s[1]
            if isinstance(d, int) and 128 <= d <= 4096 and d != 1000:
                if best is None or d > best[1]:
                    best = (o.name, d)
        if best is None:
            raise RuntimeError("模型中没有找到合适的特征输出层")
        return best

    @staticmethod
    def _preprocess(rgb_256: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(rgb_256).resize((224, 224), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)  # CHW

    def embed(self, infos) -> np.ndarray:
        """对一批 ImageInfo 提取 L2 归一化嵌入向量, 返回 (N, D) float32。"""
        n = len(infos)
        out = np.zeros((n, self.output_dim), dtype=np.float32)
        input_shape = self.session.get_inputs()[0].shape
        fixed_batch = len(input_shape) > 0 and isinstance(input_shape[0], int)
        chunk = 1 if fixed_batch else self.batch_size
        for start in range(0, n, chunk):
            batch = infos[start:start + chunk]
            xs = np.stack([self._preprocess(info.rgb_256) for info in batch])
            preds = self.session.run([self.output_name], {self.input_name: xs})[0]
            if preds.ndim == 4:  # 兜底: 若输出是特征图则全局池化
                preds = preds.mean(axis=(2, 3))
            out[start:start + len(batch)] = preds
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    @staticmethod
    def candidate_pairs(emb: np.ndarray, threshold: float,
                        top_k: int = 20) -> set:
        """余弦相似度候选: 每行取 top_k 且相似度 >= threshold。"""
        n = emb.shape[0]
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1.0)
        pairs = set()
        for i in range(n):
            row = sims[i]
            idx = np.argpartition(row, -min(top_k + 1, n))[-top_k:]
            for j in idx:
                if j > i and row[j] >= threshold:
                    pairs.add((i, j))
        return pairs
