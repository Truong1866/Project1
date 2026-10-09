#!/usr/bin/env python3
"""
YOLOv8 (detect) native Inference bằng OpenVINO – KHÔNG cần ultralytics, torch, torchvision, CUDA.

Phần tiền xử lý / hậu xử lý được port gần như nguyên bản từ ultralytics
(LetterBox, non_max_suppression, scale_boxes, clip_boxes, xywh2xyxy, Colors, Annotator.box_label),
chỉ thay phần tensor của torch bằng numpy.

Yêu cầu:
    pip install openvino opencv-python numpy
    (tùy chọn) pip install pyyaml    # để đọc tên class từ metadata.yaml

Model đầu vào (làm 1 lần, máy nào có ultralytics cũng được):
    yolo export model=yolov8n.pt format=openvino        # -> yolov8n_openvino_model/yolov8n.xml
    # hoặc  format=onnx  -> yolov8n.onnx (OpenVINO đọc trực tiếp được)
Lưu ý: export mặc định (không dùng nms=True), output có shape (1, 84, 8400).

Ví dụ:
    python yolov8_openvino.py --model yolov8n_openvino_model --source bus.jpg --device GPU
    python yolov8_openvino.py --model yolov8n.onnx --source video.mp4 --device AUTO --show
    python yolov8_openvino.py --model yolov8n.onnx --source 0 --device GPU --show     # webcam
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

# --------------------------------------------------------------------------------------
# COCO-80 (mặc định của yolov8n.pt)
# --------------------------------------------------------------------------------------
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

_GLOBAL_OV_CORE = None

# --------------------------------------------------------------------------------------
# ultralytics.utils.ops (bản numpy)
# --------------------------------------------------------------------------------------
def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    y = np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:4] / 2
    y[..., :2] = xy - wh
    y[..., 2:4] = xy + wh
    return y


def clip_boxes(boxes: np.ndarray, shape) -> np.ndarray:
    """Cắt box vào trong ảnh. shape = (h, w)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """Đưa box từ ảnh đã letterbox (img1_shape) về ảnh gốc (img0_shape). Shape = (h, w)."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS tương đương torchvision.ops.nms (bỏ box có IoU > iou_thres)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.intp)


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    nc: int = 0,
    max_det: int = 300,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> list[np.ndarray]:
    """
    Port của ultralytics.utils.ops.non_max_suppression.
    prediction: (batch, 4 + nc, num_anchors), box dạng xywh.
    Trả về list (n, 6) mỗi ảnh: [x1, y1, x2, y2, conf, cls].
    """
    assert 0 <= conf_thres <= 1 and 0 <= iou_thres <= 1
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4  # số mask coeff (0 với detect)
    mi = 4 + nc
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    prediction = prediction.transpose(0, 2, 1).copy()  # (bs, anchors, 4+nc+nm)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [np.zeros((0, 6 + nm), dtype=np.float32) for _ in range(bs)]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls, mask = x[:, :4], x[:, 4:mi], x[:, mi:]
        if multi_label:
            i, j = np.nonzero(cls > conf_thres)
            x = np.concatenate(
                [box[i], x[i, 4 + j][:, None], j[:, None].astype(np.float32), mask[i]], 1
            )
        else:
            j = cls.argmax(1)[:, None]
            conf = np.take_along_axis(cls, j, 1)
            x = np.concatenate([box, conf, j.astype(np.float32), mask], 1)[conf.ravel() > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == np.asarray(classes, dtype=np.float32)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # offset theo class (batched NMS trick)
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = nms(boxes, scores, iou_thres)[:max_det]
        output[xi] = x[keep].astype(np.float32)
    return output


# --------------------------------------------------------------------------------------
# ultralytics.data.augment.LetterBox (bản numpy/cv2)
# --------------------------------------------------------------------------------------
class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True,
                 center=True, stride=32):
        self.new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else tuple(new_shape)
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.center = center
        self.stride = stride

    def __call__(self, img: np.ndarray) -> np.ndarray:
        shape = img.shape[:2]  # (h, w)
        new_shape = self.new_shape

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:  # min rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = (int(round(dh - 0.1)), int(round(dh + 0.1))) if self.center else (0, int(round(dh + 0.1)))
        left, right = (int(round(dw - 0.1)), int(round(dw + 0.1))) if self.center else (0, int(round(dw + 0.1)))
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )


# --------------------------------------------------------------------------------------
# ultralytics.utils.plotting.Colors + Annotator.box_label (rút gọn)
# --------------------------------------------------------------------------------------
class Colors:
    _HEX = (
        "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334",
        "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF",
        "FF95C8", "FF37C7",
    )

    def __init__(self):
        self.palette = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in self._HEX]  # RGB
        self.n = len(self.palette)

    def __call__(self, i: int, bgr: bool = True):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c


COLORS = Colors()


def box_label(im: np.ndarray, box, label: str = "", color=(128, 128, 128), txt_color=(255, 255, 255), lw=None):
    lw = lw or max(round(sum(im.shape[:2]) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
            thickness=tf, lineType=cv2.LINE_AA,
        )
    return im


# --------------------------------------------------------------------------------------
# Kết quả
# --------------------------------------------------------------------------------------
@dataclass
class Detections:
    boxes: np.ndarray            # (n, 4) xyxy, pixel của ảnh gốc
    conf: np.ndarray             # (n,)
    cls: np.ndarray              # (n,) int
    names: dict[int, str]
    orig_shape: tuple[int, int]  # (h, w)
    speed: dict[str, float] = field(default_factory=dict)  # ms

    def __len__(self):
        return len(self.cls)

    def plot(self, img: np.ndarray, line_width=None) -> np.ndarray:
        img = img.copy()
        for b, c, k in zip(self.boxes, self.conf, self.cls):
            box_label(img, b, f"{self.names.get(int(k), int(k))} {c:.2f}", COLORS(int(k)), lw=line_width)
        return img


# --------------------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------------------
class YOLOv8OpenVINO:
    """
    Tương đương `DetectionPredictor` của ultralytics nhưng chạy thuần OpenVINO.

    device: "CPU", "GPU" (Intel iGPU/dGPU qua OpenCL, không cần CUDA), "GPU.0", "NPU", "AUTO", ...
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes=None,
        agnostic_nms: bool = False,
        names: dict[int, str] | list[str] | None = None,
        cache_dir: str | None = ".ov_cache",
        performance_hint: str = "LATENCY",
    ):
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.classes, self.agnostic_nms = classes, agnostic_nms

        model_file = self._resolve_model(Path(model_path))

        global _GLOBAL_OV_CORE
        if _GLOBAL_OV_CORE is None:
            _GLOBAL_OV_CORE = ov.Core()
        core = _GLOBAL_OV_CORE

        avail = core.available_devices
        base = device.split(":")[0].split(".")[0].upper()
        if base not in ("AUTO", "MULTI", "HETERO") and not any(d.startswith(base) for d in avail):
            raise RuntimeError(
                f"Device '{device}' không khả dụng. Thiết bị OpenVINO thấy được: {avail}. "
                "Với GPU Intel hãy kiểm tra đã cài intel-compute-runtime (Linux) / driver Intel Graphics (Windows)."
            )

        if cache_dir:  # cache kernel đã compile -> khởi động GPU nhanh hơn rất nhiều ở lần sau
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})

        model = core.read_model(str(model_file))

        # Nếu model có shape động thì cố định về imgsz (giữ nguyên các chiều đã tĩnh)
        pshape = model.input(0).partial_shape
        if pshape.is_dynamic:
            default = [1, 3, imgsz, imgsz]
            model.reshape([d.get_length() if d.is_static else default[i] for i, d in enumerate(pshape)])

        self.compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": performance_hint})
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        _, _, h, w = self.input_layer.shape
        self.imgsz = (int(h), int(w))
        self.device = device

        # Input tĩnh -> auto=False (giống ultralytics khi backend là OpenVINO/ONNX tĩnh)
        self.letterbox = LetterBox(self.imgsz, auto=False, stride=32)
        self.names = self._load_names(model_file, names)

    # ---- helpers ---------------------------------------------------------------------
    @staticmethod
    def _resolve_model(p: Path) -> Path:
        if p.is_dir():  # thư mục export của ultralytics: yolov8n_openvino_model/
            xmls = sorted(p.glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(f"Không thấy file .xml trong {p}")
            return xmls[0]
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    @staticmethod
    def _load_names(model_file: Path, names) -> dict[int, str]:
        if names is not None:
            return dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
        meta = model_file.parent / "metadata.yaml"
        if meta.exists():
            try:
                import yaml  # noqa: PLC0415

                data = yaml.safe_load(meta.read_text(encoding="utf-8"))
                n = data.get("names")
                if n:
                    return {int(k): v for k, v in n.items()} if isinstance(n, dict) else dict(enumerate(n))
            except Exception:  # noqa: BLE001
                pass
        return dict(enumerate(COCO80))

    # ---- pipeline (tên giống ultralytics) --------------------------------------------
    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> RGB NCHW float32 [0,1]  (giống BasePredictor.preprocess)."""
        im = self.letterbox(img_bgr)
        im = im[None, ..., ::-1].transpose((0, 3, 1, 2))  # BGR->RGB, HWC->CHW, thêm batch
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255.0
        return im

    def infer(self, im: np.ndarray) -> np.ndarray:
        return self.compiled(im)[self.output_layer]

    def postprocess(self, preds: np.ndarray, im_shape, orig_shape) -> Detections:
        """NMS + đưa box về toạ độ ảnh gốc (giống DetectionPredictor.postprocess)."""
        if preds.ndim == 3 and preds.shape[1] > preds.shape[2]:  # (1, anchors, 4+nc) -> (1, 4+nc, anchors)
            preds = preds.transpose(0, 2, 1)
        out = non_max_suppression(
            preds,
            self.conf,
            self.iou,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=len(self.names) if preds.shape[1] - 4 == len(self.names) else 0,
        )[0]
        boxes = scale_boxes(im_shape[2:], out[:, :4].copy(), orig_shape[:2])
        return Detections(
            boxes=boxes,
            conf=out[:, 4],
            cls=out[:, 5].astype(int),
            names=self.names,
            orig_shape=tuple(orig_shape[:2]),
        )

    def __call__(self, img_bgr: np.ndarray) -> Detections:
        t0 = time.perf_counter()
        im = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        preds = self.infer(im)
        t2 = time.perf_counter()
        det = self.postprocess(preds, im.shape, img_bgr.shape)
        t3 = time.perf_counter()
        det.speed = {
            "preprocess": (t1 - t0) * 1e3,
            "Inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return det

    predict = __call__


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 + OpenVINO (không cần ultralytics/CUDA)")
    ap.add_argument("--model", required=True, help=".xml / .onnx / thư mục *_openvino_model")
    ap.add_argument("--source", required=True, help="ảnh, video, hoặc số (webcam, vd 0)")
    ap.add_argument("--device", default="AUTO", help="CPU | GPU | GPU.0 | NPU | AUTO (mặc định AUTO)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640, help="chỉ dùng khi model có shape động")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="lọc theo id class")
    ap.add_argument("--agnostic-nms", action="store_true")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ (cần OpenCV có GUI)")
    ap.add_argument("--save", default=None, help="đường dẫn lưu kết quả (ảnh hoặc video .mp4)")
    return ap.parse_args()


def main():
    a = _parse_args()
    yolo = YOLOv8OpenVINO(
        a.model, a.device, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
        classes=a.classes, agnostic_nms=a.agnostic_nms,
    )
    print(f"[OpenVINO] device={a.device} | input={yolo.input_layer.shape} | devices={ov.Core().available_devices}")

    src = int(a.source) if a.source.isdigit() else a.source
    is_image = isinstance(src, str) and Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # ---- ảnh --------------------------------------------------------------------------
    if is_image:
        img = cv2.imread(src)
        if img is None:
            raise FileNotFoundError(src)
        yolo(img)  # warm-up (compile kernel GPU ở lần chạy đầu)
        det = yolo(img)
        s = det.speed
        print(f"{len(det)} object | pre {s['preprocess']:.1f} ms, infer {s['Inference']:.1f} ms, "
              f"post {s['postprocess']:.1f} ms")
        for b, c, k in zip(det.boxes, det.conf, det.cls):
            print(f"  {det.names[int(k)]:<15} {c:.2f}  {b.round(1).tolist()}")
        res = det.plot(img)
        if a.save:
            cv2.imwrite(a.save, res)
            print("Đã lưu:", a.save)
        if a.show:
            cv2.imshow("YOLOv8 OpenVINO", res)
            cv2.waitKey(0)
        return

    # ---- video / webcam ---------------------------------------------------------------
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn: {a.source}")
    writer = None
    if a.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    t_hist = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        det = yolo(frame)
        t_hist.append(time.perf_counter() - t0)
        res = det.plot(frame)
        fps_avg = 1.0 / np.mean(t_hist[-30:])
        cv2.putText(res, f"{fps_avg:.1f} FPS | {a.device}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
        if writer:
            writer.write(res)
        if a.show:
            cv2.imshow("YOLOv8 OpenVINO", res)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if t_hist:
        print(f"Trung bình: {1.0 / np.mean(t_hist[1:] or t_hist):.1f} FPS (không tính frame đầu)")


if __name__ == "__main__":
    main()