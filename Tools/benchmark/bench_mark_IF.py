#!/usr/bin/env python3
"""Benchmark các model InsightFace với pipeline giống luồng thật trong BusinessLayer.

Pipeline mô phỏng (giống pipeline._run_normal_ai):
    khung hình --YOLO (người)--> box người --(cắt 60% phía trên)--> InsightFace detect --> recognition (embedding 512-d)
  * Mỗi lượt chỉ xử lý tối đa MAX_FACES_PER_RUN người (giống face.max_per_run).
  * YOLO chạy MỘT lần cho mọi model mặt (cache box người) -> so sánh công bằng, đỡ tốn thời gian.
  * Chỉ nạp 2 module detection + recognition của từng model, nạp xong đo rồi giải phóng trước khi sang model kế.

Đầu ra:
  [A] Tốc độ & tỉ lệ bắt mặt trên video: det_ms, rec_ms, tổng ms/người, p50/p95, tỉ lệ có mặt, det_score TB.
  [B] (tuỳ chọn) Độ chuẩn nhận diện nếu có thư mục ảnh FACES_DIR/<tên_người>/*.jpg:
      sim cùng người / khác người, d', accuracy ở ngưỡng tốt nhất, rank-1 (leave-one-out).

Cấu trúc model:  Models/face_models/models/<tên_model>/*.onnx   (vd buffalo_s, buffalo_sc, buffalo_m, buffalo_l, antelopev2)

Chạy:
    python bench_mark_IF.py
    python bench_mark_IF.py --video input.mp4 --faces-dir ../../Data/faces_test
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import openvino as ov  # noqa: F401  (nạp OpenVINO TRƯỚC cv2 / onnxruntime để tránh xung đột DLL)
import cv2
import numpy as np

# ---- để import được BusinessLayer.* và yolo_native (cùng thư mục)
_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent.parent
for p in (str(ROOT), str(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from yolo_native import YOLOOpenVINO                              # noqa: E402
from BusinessLayer.Inference.face_recognizer import FaceRecognizer  # noqa: E402

# ======================================================================================
# BIẾN GLOBAL CẦN CHỈNH
# ======================================================================================
YOLO_MODEL = "yolov8n_640_openvino_model"      # tên thư mục trong Models/ (YOLO chạy để lấy box người)
FACE_MODELS = ["buffalo_s", "buffalo_m", "hyprid"]       # tên các model InsightFace cần test (thư mục trong Models/face_models/models/)

# ---- đường dẫn
MODELS_DIR = ROOT / "Models"
FACE_ROOT = MODELS_DIR / "face_models"          # root insightface (chứa models/<tên>)
VIDEO_PATH = _SCRIPT_DIR / "input.mp4"
FACES_DIR = ROOT / "Data" / "faces_test"        # (tuỳ chọn) <tên_người>/*.jpg để đo độ chuẩn nhận diện
OUT_CSV = _SCRIPT_DIR / "result" / "bench_if.csv"

# ---- thông số giống config.yaml
YOLO_DEVICE = "GPU"
PERSON_CONF = 0.30
PERSON_IOU = 0.45
FACE_DEVICE_TYPE = "GPU_FP16"
FACE_DET_SIZE = (320, 320)
FACE_DET_THRESH = 0.5
HEAD_RATIO = 0.6
MIN_FACE_PX = 36
MAX_FACES_PER_RUN = 2                           # face.max_per_run

# ---- thông số benchmark
FRAME_STRIDE = 3                                # lấy 1 khung / N khung (mô phỏng ai_fps thấp hơn fps video)
MAX_SAMPLES = 300                               # tối đa số khung dùng để đo
YOLO_WARMUP = 3


# ======================================================================================
# FaceRecognizer cho phép chọn model (bản gốc cứng "buffalo_s")
# ======================================================================================
class FaceRecognizerNamed(FaceRecognizer):
    def __init__(self, model_name: str, **kw):
        self._model_name = model_name
        super().__init__(**kw)

    def _build_app(self, providers: list, provider_options):
        from insightface.app import FaceAnalysis

        kwargs = dict(name=self._model_name, root=self.face_model_dir, providers=providers,
                      allowed_modules=["detection", "recognition"])   # chỉ nạp module cần thiết
        if provider_options is not None:
            kwargs["provider_options"] = provider_options
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        return app


class Timer:
    """Gắn vào method của sub-model để đo thời gian từng giai đoạn."""

    def __init__(self):
        self.det: list[float] = []
        self.rec: list[float] = []

    def wrap(self, app) -> None:
        det = app.models.get("detection")
        rec = app.models.get("recognition")
        if det is not None:
            orig = det.detect

            def det_wrap(*a, **k):
                t = time.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    self.det.append((time.perf_counter() - t) * 1e3)
            det.detect = det_wrap
        if rec is not None:
            orig_get = rec.get

            def rec_wrap(*a, **k):
                t = time.perf_counter()
                try:
                    return orig_get(*a, **k)
                finally:
                    self.rec.append((time.perf_counter() - t) * 1e3)
            rec.get = rec_wrap

    def clear(self):
        self.det.clear()
        self.rec.clear()


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def _mean(a):
    return float(np.mean(a)) if len(a) else 0.0


# ======================================================================================
# Bước 1: YOLO quét video một lần, cache box người
# ======================================================================================
def scan_video_with_yolo(video_path: Path):
    model_dir = MODELS_DIR / YOLO_MODEL
    if not model_dir.exists():
        raise FileNotFoundError(f"Không thấy model YOLO: {model_dir}")
    imgsz = int(YOLO_MODEL.split("_")[1]) if len(YOLO_MODEL.split("_")) > 1 and YOLO_MODEL.split("_")[1].isdigit() else 640
    print(f"=== [1] YOLO '{YOLO_MODEL}' ({YOLO_DEVICE}, imgsz={imgsz}) quét video: {video_path.name} ===")
    try:
        yolo = YOLOOpenVINO(model_dir, device=YOLO_DEVICE, imgsz=imgsz, conf=PERSON_CONF, iou=PERSON_IOU,
                            classes=[0], cache_dir=str(model_dir.parent / ".ov_cache"))
        dev = YOLO_DEVICE
    except Exception as e:  # giống InferenceEngine: GPU lỗi -> CPU
        print(f"  YOLO không chạy được trên {YOLO_DEVICE} ({e}) -> CPU")
        yolo = YOLOOpenVINO(model_dir, device="CPU", imgsz=imgsz, conf=PERSON_CONF, iou=PERSON_IOU, classes=[0])
        dev = "CPU"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    yolo(np.zeros((480, 640, 3), dtype=np.uint8))
    samples: list[tuple[int, list[list[int]]]] = []     # (chỉ số khung, [box người đã lọc theo MAX_FACES_PER_RUN])
    yolo_ms, idx, warm = [], 0, 0
    while len(samples) < MAX_SAMPLES:
        ok = cap.grab()
        if not ok:
            break
        if idx % FRAME_STRIDE == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = time.perf_counter()
            det = yolo(frame)
            dt = (time.perf_counter() - t) * 1e3
            if warm < YOLO_WARMUP:
                warm += 1
            else:
                yolo_ms.append(dt)
            boxes = []
            for b in det.boxes:
                x1, y1, x2, y2 = (int(v) for v in b)
                if x2 - x1 >= 8 and y2 - y1 >= 16:
                    boxes.append([x1, y1, x2, y2])
            boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))   # ưu tiên người lớn nhất
            samples.append((idx, boxes[:MAX_FACES_PER_RUN]))
        idx += 1
    cap.release()
    n_person = sum(len(b) for _, b in samples)
    print(f"  {len(samples)} khung lấy mẫu | {n_person} lượt (người) cần nhận diện mặt | "
          f"YOLO TB {_mean(yolo_ms):.1f} ms (p95 {_pct(yolo_ms, 95):.1f})")
    del yolo
    gc.collect()
    return samples, dict(yolo_device=dev, yolo_avg_ms=_mean(yolo_ms), yolo_p95_ms=_pct(yolo_ms, 95))


# ======================================================================================
# Bước 2: benchmark từng model mặt trên video
# ======================================================================================
def build_face(model_name: str) -> FaceRecognizerNamed:
    return FaceRecognizerNamed(model_name, face_model_dir=FACE_ROOT, det_size=FACE_DET_SIZE,
                               det_thresh=FACE_DET_THRESH, head_ratio=HEAD_RATIO, min_face_px=MIN_FACE_PX,
                               device_type=FACE_DEVICE_TYPE)


def bench_video(fr: FaceRecognizerNamed, video_path: Path, samples) -> dict:
    timer = Timer()
    timer.wrap(fr.app)
    wanted = {i: boxes for i, boxes in samples if boxes}
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    total_ms, scores, faces_per_person = [], [], []
    persons = with_face = 0
    last = max(wanted)
    idx = 0
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                for box in wanted[idx]:
                    persons += 1
                    t = time.perf_counter()
                    res = fr.extract(frame, box, head_only=True)      # đúng hàm pipeline gọi
                    total_ms.append((time.perf_counter() - t) * 1e3)
                    faces_per_person.append(len(res))
                    if res:
                        with_face += 1
                        scores.append(max(r["det_score"] for r in res))
        idx += 1
    cap.release()
    return dict(persons=persons, with_face=with_face,
                face_rate=with_face / persons if persons else 0.0,
                det_score=_mean(scores), faces_per_person=_mean(faces_per_person),
                det_ms=_mean(timer.det), det_p95=_pct(timer.det, 95),
                rec_ms=_mean(timer.rec), rec_p95=_pct(timer.rec, 95),
                total_ms=_mean(total_ms), total_p50=_pct(total_ms, 50), total_p95=_pct(total_ms, 95),
                fps_equiv=1000.0 / _mean(total_ms) if total_ms else 0.0)


# ======================================================================================
# Bước 3 (tuỳ chọn): độ chuẩn nhận diện trên thư mục ảnh <người>/*.jpg
# ======================================================================================
def bench_gallery(fr: FaceRecognizerNamed, faces_dir: Path) -> dict:
    if not faces_dir.exists():
        return {}
    embs, labels = [], []
    for person in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        for img_path in sorted(person.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res = fr.extract(img, None, head_only=False)              # ảnh chân dung: quét cả khung
            if not res:
                continue
            best = max(res, key=lambda r: (r["abs_box"][2] - r["abs_box"][0]) * (r["abs_box"][3] - r["abs_box"][1]))
            e = np.asarray(best["embedding"], np.float32)
            embs.append(e / (np.linalg.norm(e) + 1e-9))
            labels.append(person.name)
    if len(embs) < 4 or len(set(labels)) < 2:
        return {}
    E = np.vstack(embs)
    L = np.array(labels)
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    same = (L[:, None] == L[None, :])[iu]
    sims = S[iu]
    gen, imp = sims[same], sims[~same]
    if len(gen) == 0 or len(imp) == 0:
        return {}
    dprime = (gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-9)
    ths = np.linspace(-0.2, 0.9, 221)
    accs = [((gen >= t).sum() + (imp < t).sum()) / (len(gen) + len(imp)) for t in ths]
    k = int(np.argmax(accs))
    S2 = S.copy()
    np.fill_diagonal(S2, -2)
    nn = S2.argmax(1)
    rank1 = float((L[nn] == L).mean())
    return dict(g_images=len(E), g_people=len(set(labels)), sim_same=float(gen.mean()), sim_diff=float(imp.mean()),
                dprime=float(dprime), best_thr=float(ths[k]), best_acc=float(accs[k]), rank1=rank1)


# ======================================================================================
# In kết quả
# ======================================================================================
def print_tables(rows: list[dict], yolo_info: dict) -> None:
    sep = "─" * 118
    print(f"\n{'═' * 118}\n  [A] TỐC ĐỘ & TỈ LỆ BẮT MẶT (YOLO {YOLO_MODEL}: TB {yolo_info['yolo_avg_ms']:.1f} ms/khung)")
    print(f"{'═' * 118}")
    print(f"{'Model':<14}{'Dev':<10}{'Người':>6}{'Có mặt':>8}{'Rate':>7}{'detSc':>7}"
          f"{'det ms':>8}{'rec ms':>8}{'tổng TB':>9}{'p50':>7}{'p95':>7}{'≈FPS':>7}{'Mặt/ng':>8}")
    print(sep)
    for r in rows:
        if r.get("error"):
            print(f"{r['model']:<14}[LỖI] {r['error'][:90]}")
            continue
        print(f"{r['model']:<14}{r['device']:<10}{r['persons']:>6}{r['with_face']:>8}{r['face_rate']:>7.2f}"
              f"{r['det_score']:>7.2f}{r['det_ms']:>8.1f}{r['rec_ms']:>8.1f}{r['total_ms']:>9.1f}"
              f"{r['total_p50']:>7.1f}{r['total_p95']:>7.1f}{r['fps_equiv']:>7.1f}{r['faces_per_person']:>8.2f}")
    print(sep)
    g = [r for r in rows if r.get("g_images")]
    if g:
        print(f"\n{'═' * 118}\n  [B] ĐỘ CHUẨN NHẬN DIỆN ({FACES_DIR})\n{'═' * 118}")
        print(f"{'Model':<14}{'Ảnh':>6}{'Người':>7}{'sim cùng':>10}{'sim khác':>10}{'d′':>7}{'ngưỡng*':>9}{'acc*':>7}{'rank-1':>8}")
        print(sep)
        for r in g:
            print(f"{r['model']:<14}{r['g_images']:>6}{r['g_people']:>7}{r['sim_same']:>10.3f}{r['sim_diff']:>10.3f}"
                  f"{r['dprime']:>7.2f}{r['best_thr']:>9.2f}{r['best_acc']:>7.3f}{r['rank1']:>8.3f}")
        print(sep)
        print("  d′ càng cao càng tốt (tách người quen/lạ); ngưỡng* = ngưỡng cosine cho accuracy cao nhất "
              "(so với face.threshold trong config.yaml).")
    ok = [r for r in rows if not r.get("error") and r["persons"]]
    if ok:
        fast = min(ok, key=lambda r: r["total_ms"])
        print(f"\n  ⚡ Nhanh nhất: {fast['model']} ({fast['total_ms']:.1f} ms/người)")
        rate = max(ok, key=lambda r: r["face_rate"])
        print(f"  👤 Bắt mặt nhiều nhất: {rate['model']} ({rate['face_rate']:.2f})")
        if g:
            best = max(g, key=lambda r: r["dprime"])
            print(f"  🏆 Tách người tốt nhất (d′): {best['model']} ({best['dprime']:.2f})")


def save_csv(rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu CSV: {OUT_CSV}")


# ======================================================================================
def main() -> int:
    global YOLO_MODEL, FACE_MODELS, FACES_DIR
    ap = argparse.ArgumentParser(description="Benchmark InsightFace với pipeline YOLO -> mặt")
    ap.add_argument("--video", default=str(VIDEO_PATH))
    ap.add_argument("--yolo", default=YOLO_MODEL)
    ap.add_argument("--face-models", nargs="*", default=FACE_MODELS)
    ap.add_argument("--faces-dir", default=str(FACES_DIR))
    a = ap.parse_args()
    YOLO_MODEL, FACE_MODELS, FACES_DIR = a.yolo, list(a.face_models), Path(a.faces_dir)

    video = Path(a.video)
    if not video.exists():
        print(f"Lỗi: không thấy video '{video}'")
        return 1

    samples, yolo_info = scan_video_with_yolo(video)
    if not any(b for _, b in samples):
        print("YOLO không thấy người nào trong video -> không có gì để đo mặt.")
        return 1

    rows: list[dict] = []
    for i, name in enumerate(FACE_MODELS, 1):
        print(f"\n=== [2.{i}] InsightFace '{name}' ===")
        mdir = FACE_ROOT / "models" / name
        if not mdir.exists():
            print(f"  Bỏ qua: không thấy {mdir}")
            rows.append(dict(model=name, error=f"không thấy {mdir}"))
            continue
        fr = None
        try:
            t0 = time.perf_counter()
            fr = build_face(name)
            load_s = time.perf_counter() - t0
            row = dict(model=name, device=fr.device_label, load_s=round(load_s, 2),
                       modules="+".join(fr.app.models.keys()))
            print(f"  nạp {load_s:.2f}s | {row['device']} | modules: {row['modules']}")
            row.update(bench_video(fr, video, samples))
            row.update(bench_gallery(fr, FACES_DIR))
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append(dict(model=name, error=f"{type(e).__name__}: {e}"))
        finally:
            del fr
            gc.collect()
            time.sleep(0.5)

    print_tables(rows, yolo_info)
    save_csv([{**r, **yolo_info} for r in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())