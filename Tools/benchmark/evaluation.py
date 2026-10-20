"""
evaluation.py – Benchmark độ chuẩn (mAP) của tất cả model YOLO trong thư mục Models/.

Công cụ này tự động:
  1. Quét thư mục ../../Models/ để tìm tất cả model YOLO hợp lệ
     (OpenVINO folder: *_openvino_model, hoặc file .pt / .onnx).
  2. Chạy validation với dataset coco128.yaml (cùng thư mục với file này).
  3. Tổng hợp và in bảng so sánh mAP@0.5 và mAP@0.5:0.95.

Chạy:
    python evaluation.py
    python evaluation.py --models-dir /đường/dẫn/khác --device cuda:0
"""

import os
import sys
import time
import gc
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Cấu hình mặc định
# ---------------------------------------------------------------------------

# Thư mục Models tương đối so với vị trí file này
_SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = _SCRIPT_DIR / "../../Models/yolov8l_640_openvino_model"

# Dataset để đánh giá (cùng thư mục với file này)
DATASET_YAML = str(_SCRIPT_DIR / "coco128.yaml")

# Thiết bị mặc định để chạy val (cpu để ổn định, không cần GPU)
DEFAULT_DEVICE = "cpu"

# Ngưỡng conf & iou khi val (để nhất quán với bench_mark.py)
CONF = 0.5
IOU = 0.4

# Class id cần lấy kết quả riêng biệt (0 = person trong COCO)
PERSON_CLASS_ID = 0


# ---------------------------------------------------------------------------
# Cấu trúc lưu kết quả
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    model_name: str
    imgsz: int
    map50_all: float = 0.0        # mAP@0.5 tất cả class
    map_all: float = 0.0          # mAP@0.5:0.95 tất cả class
    map50_person: float = 0.0     # mAP@0.5 riêng lớp Person
    map_person: float = 0.0       # mAP@0.5:0.95 riêng lớp Person
    precision: float = 0.0
    recall: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Phát hiện model
# ---------------------------------------------------------------------------

def _get_imgsz_from_path(path: str) -> int:
    """Trích image size từ tên folder/file, ví dụ 'yolov8n_640_openvino_model' → 640."""
    parts = Path(path).name.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 640  # fallback

def prepare_model(models_dir: Path):
    if not models_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục Models: {models_dir}")
        return None
    imgsz = _get_imgsz_from_path(models_dir.name)
    result = [str(models_dir), imgsz]
    return result



def discover_models(models_dir: Path) -> list[tuple[str, int]]:
    """
    Trả về danh sách (model_path, imgsz) cho tất cả model YOLO tìm thấy.

    Thứ tự ưu tiên phát hiện:
      1. Thư mục con có hậu tố '_openvino_model' (OpenVINO IR).
      2. File .pt (PyTorch weight).
      3. File .onnx (ONNX).
    """
    models_dir = models_dir.resolve()
    if not models_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục Models: {models_dir}")
        return []

    found: list[tuple[str, int]] = []

    # 1) OpenVINO IR folders
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and entry.name.endswith("_openvino_model"):
            imgsz = _get_imgsz_from_path(entry.name)
            found.append((str(entry), imgsz))

    # 2) .pt files (bỏ qua nếu đã có bản OpenVINO tương ứng)
    for entry in sorted(models_dir.glob("*.pt")):
        found.append((str(entry), 640))

    # 3) .onnx files
    for entry in sorted(models_dir.glob("*.onnx")):
        imgsz = _get_imgsz_from_path(entry.name)
        found.append((str(entry), imgsz))

    return found


# ---------------------------------------------------------------------------
# Đánh giá một model
# ---------------------------------------------------------------------------

def evaluate_one(model_path: str, imgsz: int, dataset_yaml: str, device: str) -> EvalResult:
    """Chạy validation Ultralytics cho một model và trả về EvalResult."""
    model_name = Path(model_path).name
    print(f"\n{'='*60}")
    print(f"  MODEL : {model_name}")
    print(f"  IMGSZ : {imgsz}  |  DEVICE: {device.upper()}")
    print(f"{'='*60}")

    result = EvalResult(model_name=model_name, imgsz=imgsz)
    t0 = time.perf_counter()

    try:
        from ultralytics import YOLO
    except ImportError:
        result.error = "Thiếu thư viện 'ultralytics'. Hãy chạy: pip install ultralytics"
        print(f"[LỖI] {result.error}")
        return result

    try:
        model = YOLO(model_path, task="detect")

        val_results = model.val(
            data=dataset_yaml,
            imgsz=imgsz,
            conf=CONF,
            iou=IOU,
            device=device,
            verbose=False,   # tắt log dài dòng
            plots=False,     # không lưu ảnh kết quả
        )

        result.map50_all = float(val_results.box.map50)
        result.map_all   = float(val_results.box.map)
        result.precision = float(val_results.box.mp)   # mean precision
        result.recall    = float(val_results.box.mr)   # mean recall

        # Lấy riêng kết quả lớp Person (class id 0)
        ap_class_index = list(val_results.box.ap_class_index)
        if PERSON_CLASS_ID in ap_class_index:
            idx = ap_class_index.index(PERSON_CLASS_ID)
            result.map50_person = float(val_results.box.ap50[idx])
            result.map_person   = float(val_results.box.ap[idx])

        result.elapsed_sec = time.perf_counter() - t0

        # In tóm tắt ngay sau khi chạy xong
        print(f"  ✔ Tất cả class  → mAP@0.5: {result.map50_all:.4f} | mAP@0.5:0.95: {result.map_all:.4f}")
        print(f"  ✔ Lớp Person    → mAP@0.5: {result.map50_person:.4f} | mAP@0.5:0.95: {result.map_person:.4f}")
        print(f"  ✔ Precision: {result.precision:.4f} | Recall: {result.recall:.4f}")
        print(f"  ✔ Thời gian val: {result.elapsed_sec:.1f}s")

    except Exception as exc:
        result.elapsed_sec = time.perf_counter() - t0
        result.error = str(exc)
        print(f"[LỖI] Khi đánh giá '{model_name}': {exc}")

    finally:
        # Giải phóng bộ nhớ giữa các model
        try:
            del model
        except Exception:
            print("Can't delete model by some reason")
        gc.collect()
        time.sleep(0.5)

    return result


# ---------------------------------------------------------------------------
# In bảng tổng hợp
# ---------------------------------------------------------------------------

_COL_W = {
    "model": 38,
    "sz": 6,
    "map50_all": 12,
    "map_all": 14,
    "map50_p": 12,
    "map_p": 14,
    "prec": 10,
    "rec": 10,
    "time": 10,
}


def _header() -> str:
    w = _COL_W
    return (
        f"{'Model':<{w['model']}} "
        f"{'Sz':>{w['sz']}} "
        f"{'mAP@.5':>{w['map50_all']}} "
        f"{'mAP@.5:.95':>{w['map_all']}} "
        f"{'P-mAP@.5':>{w['map50_p']}} "
        f"{'P-mAP@.5:.95':>{w['map_p']}} "
        f"{'Precision':>{w['prec']}} "
        f"{'Recall':>{w['rec']}} "
        f"{'Time(s)':>{w['time']}}"
    )


def _row(r: EvalResult) -> str:
    w = _COL_W
    if r.error:
        label = f"[LỖI] {r.error[:40]}"
        return f"{r.model_name:<{w['model']}} {r.imgsz:>{w['sz']}} {label}"
    return (
        f"{r.model_name:<{w['model']}} "
        f"{r.imgsz:>{w['sz']}} "
        f"{r.map50_all:>{w['map50_all']}.4f} "
        f"{r.map_all:>{w['map_all']}.4f} "
        f"{r.map50_person:>{w['map50_p']}.4f} "
        f"{r.map_person:>{w['map_p']}.4f} "
        f"{r.precision:>{w['prec']}.4f} "
        f"{r.recall:>{w['rec']}.4f} "
        f"{r.elapsed_sec:>{w['time']}.1f}"
    )


def print_summary(results: list[EvalResult]) -> None:
    sep = "─" * (sum(_COL_W.values()) + len(_COL_W))
    print(f"\n\n{'═'*len(sep)}")
    print("  BẢNG TỔNG HỢP ĐỘ CHUẨN (mAP) CÁC MODEL YOLO")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"{'═'*len(sep)}")
    print(_header())
    print(sep)
    for r in results:
        print(_row(r))
    print(sep)

    # Tìm model tốt nhất theo mAP@0.5:0.95 (all class)
    valid = [r for r in results if r.error is None]
    if valid:
        best = max(valid, key=lambda r: r.map_all)
        print(f"\n  🏆 Model tốt nhất (mAP@0.5:0.95): {best.model_name}  ({best.map_all:.4f})")

    print(f"{'═'*len(sep)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark độ chuẩn mAP của các model YOLO trên coco128"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=MODELS_DIR,
        help=f"Thư mục chứa model (mặc định: {MODELS_DIR})"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_YAML,
        help=f"Đường dẫn tới file dataset YAML (mặc định: {DATASET_YAML})"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help="Thiết bị chạy val: cpu | cuda:0 | 0 (mặc định: cpu)"
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Chỉ đánh giá các model có tên chứa chuỗi này (lọc theo tên folder/file)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  ĐÁNH GIÁ ĐỘ CHUẨN MODEL YOLO – COCO128")
    print("═" * 60)
    print(f"  Thư mục Models : {args.models_dir.resolve()}")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Device         : {args.device.upper()}")
    if args.models:
        print(f"  Lọc model      : {args.models}")
    print()

    # Kiểm tra dataset
    if not Path(args.dataset).exists():
        print(f"[LỖI] Không tìm thấy file dataset: {args.dataset}")
        sys.exit(1)

    # Phát hiện model
    '''all_models = discover_models(args.models_dir)
    if not all_models:
        print("[LỖI] Không tìm thấy model nào trong thư mục Models.")
        sys.exit(1)

    # Lọc nếu người dùng chỉ định --models
    if args.models:
        keywords = [kw.lower() for kw in args.models]
        all_models = [
            (p, s) for p, s in all_models
            if any(kw in Path(p).name.lower() for kw in keywords)
        ]
        if not all_models:
            print(f"[LỖI] Không có model nào khớp với bộ lọc: {args.models}")
            sys.exit(1)

    print(f"  Tổng số model sẽ đánh giá: {len(all_models)}")
    for p, s in all_models:
        print(f"    • {Path(p).name}  (imgsz={s})")'''

    # Chạy đánh giá từng model
    results: list[EvalResult] = []
    model_path, imgsz = prepare_model(MODELS_DIR)
    r = evaluate_one(
        model_path=model_path,
        imgsz=imgsz,
        dataset_yaml=args.dataset,
        device=args.device,
    )
    results.append(r)

    # In bảng tổng hợp
    print_summary(results)


if __name__ == "__main__":
    main()

