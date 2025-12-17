import os
import re
import glob
import argparse
import pickle
import shutil
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import imageio.v2 as imageio
from PIL import Image
import torch

# SAM3 (per README)
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from tqdm import tqdm

# add your HF token and proxy here if needed

def to_cpu_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return [to_cpu_numpy(v) for v in x]
    return np.asarray(x)

# -------------------------
# I/O helpers
# -------------------------
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def ensure_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Your wrist frames are often shaped like (1,H,W,3) or (H,W,3).
    This normalizes to (H,W,3) uint8 RGB.
    """
    arr = np.asarray(frame)

    # squeeze leading singleton dims like (1,H,W,3)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected frame shape: {arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # convert float [0,1] -> uint8
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    return arr

import os
import numpy as np
import imageio.v2 as imageio
import cv2

def write_mp4(frames_rgb, out_path, fps=30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    assert len(frames_rgb) > 0
    # import ipdb;ipdb.set_trace()
    h, w = frames_rgb[0].shape[:2]

    # mp4v is widely available; if you have h264 you can try 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))

    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")

    for fr in frames_rgb:
        fr = np.asarray(fr)
        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)
        if fr.ndim == 2:
            fr = np.repeat(fr[:, :, None], 3, axis=2)
        if fr.shape[2] == 4:
            fr = fr[:, :, :3]
        bgr = fr[:, :, ::-1]  # RGB->BGR
        vw.write(bgr)

    vw.release()

def write_video(frames_rgb_u8, out_mp4, fps=30):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    write_mp4(frames_rgb_u8, out_mp4, fps)

def write_mask_video(masks_u8, out_mp4, fps=30):
    # Convert (H,W) -> (H,W,3)
    frames = [np.stack([m, m, m], axis=-1).astype(np.uint8) for m in masks_u8]
    write_video(frames, out_mp4, fps=fps)

def write_overlay_video(frames_rgb_u8, masks_u8, out_mp4, fps=30, alpha=0.5):
    out_frames = []
    for fr, m in zip(frames_rgb_u8, masks_u8):
        frf = fr.astype(np.float32)
        mm = (m > 0)[:, :, None].astype(np.float32)

        overlay = frf.copy()
        overlay[..., 1] = np.clip(overlay[..., 1] + 180.0 * mm[..., 0], 0, 255)

        out = (frf * (1 - alpha * mm) + overlay * (alpha * mm)).clip(0, 255).astype(np.uint8)
        out_frames.append(out)

    write_video(out_frames, out_mp4, fps=fps)



# -------------------------
# Transition structure helpers
# -------------------------
def list_transition_pkls(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    # directory: grab transitions*.pkl
    pkls = sorted(glob.glob(os.path.join(input_path, "transitions*.pkl")))
    return pkls

def list_transition_pkls_batch(folder: str, prefix: str = "transitions_", suffix: str = ".pkl") -> List[str]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    hits = []
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if m:
            hits.append((int(m.group(1)), fn))
    hits.sort(key=lambda x: x[0])
    return [os.path.join(folder, fn) for _, fn in hits]

def extract_camera_frames(transitions: List[Dict[str, Any]], cam_key: str, device="cuda") -> List[np.ndarray]:
    frames = []
    for t in transitions:
        obs = t.get("observations", {})
        if cam_key not in obs:
            raise KeyError(f"Missing observations['{cam_key}'] in a transition.")
        if device == 'cpu':
            frames.append(ensure_uint8_rgb(obs[cam_key]))
        else:
            frame_tensor = torch.from_numpy(ensure_uint8_rgb(obs[cam_key])).to(device)
            frames.append(frame_tensor)

    t_last = transitions[-1]
    obs = t.get("next_observations", {})
    if cam_key not in obs:
        raise KeyError(f"Missing observations['{cam_key}'] in a transition.")
    if device == 'cpu':
        frames.append(ensure_uint8_rgb(obs[cam_key]))
    else:
        frame_tensor = torch.from_numpy(ensure_uint8_rgb(obs[cam_key])).to(device)
        frames.append(frame_tensor)
    return frames

def insert_camera_masks(transitions: List[Dict[str, Any]], cam_key: str, masks_u8: List[np.ndarray], out_key_suffix: str = "_sam3_mask"):
    """
    Stores masks back into transitions[i]["observations"][f"{cam_key}{suffix}"] as uint8 (H,W).
    """
    if abs(len(transitions) - len(masks_u8)) > 1:
        raise ValueError(f"len(transitions)={len(transitions)} != len(masks)={len(masks_u8)}")

    out_key = f"{cam_key}{out_key_suffix}"
    for i, t in enumerate(transitions):
        if i < len(masks_u8):
            t.setdefault("observations", {})[out_key] = masks_u8[i]
            t.setdefault("next_observations", {})[out_key] = masks_u8[i+1]


# -------------------------
# SAM3 image segmentation
# -------------------------
def choose_mask_from_output(output, h, w, score_thresh=0.0, fallback="zeros"):
    masks = output.get("masks", None)
    scores = output.get("scores", None)

    masks_np = to_cpu_numpy(masks)
    scores_np = to_cpu_numpy(scores)

    # handle missing / empty
    if masks_np is None or scores_np is None:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    if scores_np.size == 0:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    # Normalize masks to shape (N,H,W)
    masks_np = np.asarray(masks_np)
    if masks_np.size == 0:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    # common cases: (N,1,H,W) or (N,H,W,1)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
        masks_np = masks_np[..., 0]

    if masks_np.ndim != 3:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    n = masks_np.shape[0]
    if n == 0:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    # score/mask length mismatch guard
    if scores_np.shape[0] != n:
        scores_np = scores_np[:n] if scores_np.shape[0] > n else np.pad(
            scores_np, (0, n - scores_np.shape[0]), constant_values=-1e9
        )

    keep = np.where(scores_np >= float(score_thresh))[0]
    if keep.size == 0:
        return np.zeros((h, w), dtype=np.uint8) if fallback == "zeros" else None

    best = int(keep[np.argmax(scores_np[keep])])

    m = masks_np[best]
    m = (m > 0).astype(np.uint8) * 255

    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    return m


class Sam3ImageSegmenter:
    def __init__(self, device="cuda", use_amp=True):
        self.device = device
        self.use_amp = use_amp

        model = build_sam3_image_model()
        model = model.to(device).eval()
        self.processor = Sam3Processor(model)
    def segment_frames_microbatched(
        self,
        frames_rgb_u8: List[np.ndarray],
        prompts: List[str],
        batch_size: int = 32,
        score_thresh: float = 0.0,
        desc: str = "SAM3 segment",
    ) -> List[np.ndarray]:
        import time
        import threading
        from tqdm import tqdm

        def _run_tqdm_timer(pbar: tqdm, t0: float, stop_evt: threading.Event, every_s: float = 0.2):
            # refresh the same tqdm line periodically (no extra prints)
            while not stop_evt.is_set():
                # elapsed_min = (time.time() - t0) / 60.0
                # pbar.set_postfix_str(f"elapsed={elapsed_min:.2f} min", refresh=False)
                pbar.refresh()
                stop_evt.wait(every_s)

        if isinstance(prompts, str):
            prompts = [prompts]

        H, W = frames_rgb_u8[0].shape[:2]
        out_masks: List[np.ndarray] = []

        n = len(frames_rgb_u8)

        # tqdm over batches, but show a constantly-updating elapsed time
        pbar = tqdm(range(0, n, batch_size), desc=desc, unit="batch", dynamic_ncols=True)
        t0 = time.time()
        stop_evt = threading.Event()
        timer_th = threading.Thread(target=_run_tqdm_timer, args=(pbar, t0, stop_evt, 0.2), daemon=True)
        timer_th.start()

        try:
            # iterate “micro-batches” but still do set_image per-frame (SAM3 API limitation)
            for s in pbar:
                e = min(s + batch_size, n)
                chunk = frames_rgb_u8[s:e]

                with torch.inference_mode():
                    amp_ctx = (
                        torch.autocast("cuda", dtype=torch.float16)
                        if (self.device.startswith("cuda") and self.use_amp)
                        else torch.no_grad()
                    )
                    with amp_ctx:
                        for fr in chunk:
                            # set_image accepts PIL or tensor. PIL is simplest & matches README.
                            state = self.processor.set_image(Image.fromarray(fr))

                            union = np.zeros((H, W), dtype=np.uint8)
                            for p in prompts:
                                output = self.processor.set_text_prompt(state=state, prompt=p)
                                m = choose_mask_from_output(
                                    output, h=H, w=W, score_thresh=score_thresh, fallback="zeros"
                                )
                                union = np.maximum(union, m)

                            out_masks.append(union)
        finally:
            stop_evt.set()
            timer_th.join(timeout=1.0)
            pbar.close()

        return out_masks

def parse_prompts(prompt: str) -> List[str]:
    """
    Allow multiple prompts joined with |, e.g. "peg|hole"
    """
    parts = [p.strip() for p in prompt.split("|")]
    return [p for p in parts if p]

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)

def segment_frames_with_sam3_image_model(frames_rgb_u8, prompts, score_thresh=0.0, device="cuda"):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    # allow prompts to be str or list[str]
    if isinstance(prompts, str):
        prompts = [prompts]

    H, W = frames_rgb_u8[0].shape[:2]
    masks_out = []

    for fr in tqdm(frames_rgb_u8, desc="SAM3 segmenting", unit="frame"):
        image = Image.fromarray(fr)  # RGB
        state = processor.set_image(image)

        union = np.zeros((H, W), dtype=np.uint8)
        for p in prompts:
            output = processor.set_text_prompt(state=state, prompt=p)
            m = choose_mask_from_output(output, h=H, w=W, score_thresh=score_thresh, fallback="zeros")
            union = np.maximum(union, m)

        masks_out.append(union)

    return masks_out  # list of uint8 HxW (0/255)

# SAM3 video segmentation helper
import math

def mask_union_from_output(output, h, w, score_thresh=0.0) -> np.ndarray:
    masks = output.get("masks", None)
    scores = output.get("scores", None)

    masks_np = to_cpu_numpy(masks)
    scores_np = to_cpu_numpy(scores)

    if masks_np is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks_np = np.asarray(masks_np)
    # normalize to (N,H,W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
        masks_np = masks_np[..., 0]
    if masks_np.ndim != 3 or masks_np.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    n = masks_np.shape[0]

    if scores_np is None:
        keep = np.arange(n)
    else:
        scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)
        if scores_np.size != n:
            scores_np = scores_np[:n] if scores_np.size > n else np.pad(scores_np, (0, n - scores_np.size), constant_values=-1e9)
        keep = np.where(scores_np >= float(score_thresh))[0]
        if keep.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

    union = (masks_np[keep] > 0).any(axis=0).astype(np.uint8) * 255
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union


def _frames_to_jpeg_folder(frames_rgb_u8, out_dir, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    # write only if missing (unless overwrite)
    for i, fr in enumerate(frames_rgb_u8):
        fp = os.path.join(out_dir, f"{i:06d}.jpg")
        if (not overwrite) and os.path.exists(fp):
            continue
        Image.fromarray(fr).save(fp, quality=95)

def _make_chunk_view_folder(src_jpeg_dir, dst_dir, s, e, overwrite=True):
    """
    Create dst_dir containing frames [s,e) as 000000.jpg.. in order.
    Uses hardlinks when possible (fast, no extra storage).
    """
    if overwrite and os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    j = 0
    for idx in range(s, e):
        src = os.path.join(src_jpeg_dir, f"{idx:06d}.jpg")
        dst = os.path.join(dst_dir, f"{j:06d}.jpg")
        j += 1
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing source JPEG: {src}")

        try:
            os.link(src, dst)  # hardlink
        except OSError:
            # fallback: symlink (still cheap)
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

def _choose_mask_any_output(output, h, w, score_thresh=0.0):
    """
    More tolerant than your choose_mask_from_output:
    - scores may be missing
    - output may have masks as (N,H,W) or (H,W) etc
    Returns uint8 (H,W) in {0,255}.
    """
    masks = output.get("masks", None)
    scores = output.get("scores", None)

    # convert to CPU numpy
    if hasattr(masks, "detach"):
        masks = masks.detach().float().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().float().cpu().numpy()

    masks = None if masks is None else np.asarray(masks)
    scores = None if scores is None else np.asarray(scores).reshape(-1)

    if masks is None or masks.size == 0:
        return np.zeros((h, w), np.uint8)

    # normalize masks to (N,H,W)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]

    if masks.ndim != 3 or masks.shape[0] == 0:
        return np.zeros((h, w), np.uint8)

    n = masks.shape[0]
    if scores is None or scores.size == 0:
        best = 0
    else:
        if scores.shape[0] != n:
            scores = scores[:n] if scores.shape[0] > n else np.pad(scores, (0, n - scores.shape[0]), constant_values=-1e9)
        keep = np.where(scores >= float(score_thresh))[0]
        if keep.size == 0:
            return np.zeros((h, w), np.uint8)
        best = int(keep[np.argmax(scores[keep])])

    m = masks[best]
    if m.shape != (h, w):
        # conservative: if shapes mismatch badly, return zeros
        return np.zeros((h, w), np.uint8)

    return ((m > 0).astype(np.uint8) * 255)

def _extract_framewise_masks_from_outputs(outputs, T, H, W, score_thresh=0.0):
    """
    Tries common patterns:
    A) outputs is list length T, each entry dict with masks/scores
    B) outputs is dict with "masks" as (T,...) tensor/array
    """
    if outputs is None:
        return None

    # Pattern A
    if isinstance(outputs, list) and len(outputs) == T:
        out = []
        for o in outputs:
            if not isinstance(o, dict):
                return None
            out.append(_choose_mask_any_output(o, H, W, score_thresh=score_thresh))
        return out

    # Pattern B
    if isinstance(outputs, dict) and "masks" in outputs:
        masks = outputs["masks"]
        if hasattr(masks, "detach"):
            masks = masks.detach().float().cpu().numpy()
        masks = np.asarray(masks)

        # try (T,1,H,W) -> (T,H,W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 3 and masks.shape[0] == T and masks.shape[1] == H and masks.shape[2] == W:
            return [((masks[i] > 0).astype(np.uint8) * 255) for i in range(T)]

    return None

def segment_video_predictor_num_chunks(
    video_predictor,
    frames_rgb_u8: List[np.ndarray],
    prompts: List[str],
    tmp_root: str,
    score_thresh: float = 0.0,
    num_chunks: int = 4,
    overlap: int = 8,
    anchor_freq: int = 0,
) -> List[np.ndarray]:
    T = len(frames_rgb_u8)
    H, W = frames_rgb_u8[0].shape[:2]
    out = [np.zeros((H, W), dtype=np.uint8) for _ in range(T)]

    num_chunks = max(1, int(num_chunks))
    base = int(np.ceil(T / num_chunks))

    for ci in range(num_chunks):
        s0 = ci * base
        e0 = min(T, (ci + 1) * base)
        if s0 >= T:
            break

        s = max(0, s0 - (overlap if ci > 0 else 0))
        e = min(T, e0 + (overlap if ci < num_chunks - 1 else 0))

        chunk = frames_rgb_u8[s:e]
        chunk_tmp = os.path.join(tmp_root, f"chunk_{ci:03d}_{s:06d}_{e:06d}")

        try:
            cm = segment_video_predictor_one_chunk(
                video_predictor=video_predictor,
                frames_rgb_u8=chunk,
                prompts=prompts,
                tmp_dir=chunk_tmp,
                score_thresh=score_thresh,
                anchor_freq=anchor_freq,
            )
        except Exception as ex:
            raise RuntimeError(f"SAM3 video predictor failed for chunk {ci} [{s},{e}): {type(ex).__name__}: {ex}")

        # merge: union in overlap
        for j, m in enumerate(cm):
            gi = s + j
            out[gi] = np.maximum(out[gi], m)

    return out




def _pick_best_mask_from_arrays(masks_np, scores_np, h, w, score_thresh=0.0):
    """
    masks_np: (N,H,W) float/bool/int
    scores_np: (N,) float
    returns uint8 (H,W) in {0,255}
    """
    if masks_np is None or scores_np is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks_np = np.asarray(masks_np)
    scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)

    if masks_np.size == 0 or scores_np.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # normalize masks to (N,H,W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 4 and masks_np.shape[-1] == 1:
        masks_np = masks_np[..., 0]
    if masks_np.ndim != 3:
        return np.zeros((h, w), dtype=np.uint8)

    n = masks_np.shape[0]
    if scores_np.shape[0] != n:
        scores_np = scores_np[:n] if scores_np.shape[0] > n else np.pad(scores_np, (0, n - scores_np.shape[0]), constant_values=-1e9)

    keep = np.where(scores_np >= float(score_thresh))[0]
    if keep.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    best = int(keep[np.argmax(scores_np[keep])])
    m = masks_np[best]
    m = (m > 0).astype(np.uint8) * 255

    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def _extract_masks_from_video_outputs(outputs, T, h, w, score_thresh=0.0):
    """
    Best-effort extraction of T frame masks from SAM3 video outputs.
    Returns List[np.uint8] length T or None if can't parse.
    """
    # Case 1: list of per-frame dicts
    if isinstance(outputs, list) and len(outputs) == T and all(isinstance(o, dict) for o in outputs):
        out = []
        for o in outputs:
            m = choose_mask_from_output(o, h=h, w=w, score_thresh=score_thresh, fallback="zeros")
            out.append(m)
        return out

    # Case 2: dict with masks (maybe time-first)
    if isinstance(outputs, dict) and "masks" in outputs:
        masks = to_cpu_numpy(outputs.get("masks"))
        scores = to_cpu_numpy(outputs.get("scores"))

        if masks is None:
            return None

        masks = np.asarray(masks)

        # Try common shapes:
        # (T,H,W) -> already per-frame binary mask
        if masks.ndim == 3 and masks.shape[0] == T:
            return [((masks[i] > 0).astype(np.uint8) * 255) for i in range(T)]

        # (T,1,H,W)
        if masks.ndim == 4 and masks.shape[0] == T and masks.shape[1] == 1:
            return [((masks[i, 0] > 0).astype(np.uint8) * 255) for i in range(T)]

        # (T,N,H,W) with (T,N) scores
        if masks.ndim == 4 and masks.shape[0] == T:
            if scores is None:
                # no scores -> pick first mask each frame
                return [((masks[i, 0] > 0).astype(np.uint8) * 255) for i in range(T)]

            scores = np.asarray(scores, dtype=np.float32)
            if scores.ndim == 2 and scores.shape[0] == T:
                out = []
                for i in range(T):
                    out.append(_pick_best_mask_from_arrays(masks[i], scores[i], h, w, score_thresh))
                return out

    return None
# def _frames_to_jpeg_folder(frames_rgb_u8: List[np.ndarray], tmp_dir: str):
#     os.makedirs(tmp_dir, exist_ok=True)
#     for i, fr in enumerate(frames_rgb_u8):
#         Image.fromarray(fr).save(os.path.join(tmp_dir, f"{i:06d}.jpg"), quality=95)

def _propagate_outputs_per_frame(video_predictor, session_id: str):
    outputs_per_frame = {}

    # Most SAM3 builds expose a streaming API for propagation
    if hasattr(video_predictor, "handle_stream_request"):
        stream = video_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        )
        for resp in stream:
            fi = resp.get("frame_index", None)
            out = resp.get("outputs", None)
            if fi is None or out is None:
                continue
            outputs_per_frame[int(fi)] = out
        return outputs_per_frame

    # Fallback: some builds may return everything in one response
    resp = video_predictor.handle_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    )
    # Try to interpret common shapes
    if isinstance(resp, list):
        for r in resp:
            fi = r.get("frame_index", None)
            out = r.get("outputs", None)
            if fi is not None and out is not None:
                outputs_per_frame[int(fi)] = out
    elif isinstance(resp, dict) and "outputs_per_frame" in resp:
        return resp["outputs_per_frame"]

    return outputs_per_frame

def segment_video_predictor_one_chunk(
    video_predictor,
    frames_rgb_u8: List[np.ndarray],
    prompts: List[str],
    tmp_dir: str,
    score_thresh: float = 0.0,
    anchor_freq: int = 0,   # 0 disables extra anchors
) -> List[np.ndarray]:
    H, W = frames_rgb_u8[0].shape[:2]
    T = len(frames_rgb_u8)

    # write JPEGs and start a predictor session on that folder
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _frames_to_jpeg_folder(frames_rgb_u8, tmp_dir)

    resp = video_predictor.handle_request(
        request=dict(type="start_session", resource_path=tmp_dir)
    )
    sid = resp.get("session_id", None)
    if sid is None:
        raise RuntimeError("start_session did not return session_id")

    # add prompts (optionally at anchor frames too)
    anchor_frames = [0]
    if anchor_freq and anchor_freq > 0:
        anchor_frames += list(range(anchor_freq, T, anchor_freq))

    for p in prompts:
        for fi in anchor_frames:
            video_predictor.handle_request(
                request=dict(type="add_prompt", session_id=sid, frame_index=int(fi), text=p)
            )

    outputs_per_frame = _propagate_outputs_per_frame(video_predictor, sid)
    if not outputs_per_frame:
        raise RuntimeError("propagate_in_video produced no outputs")

    masks_u8 = []
    for i in range(T):
        out = outputs_per_frame.get(i, None)
        if out is None:
            masks_u8.append(np.zeros((H, W), dtype=np.uint8))
        else:
            masks_u8.append(mask_union_from_output(out, h=H, w=W, score_thresh=score_thresh))

    return masks_u8


def segment_video_with_sam3_video_predictor(
    resource_path: str,
    frames_rgb_u8: List[np.ndarray],
    prompts: List[str],
    score_thresh: float = 0.0,
    anchor_freq: int = 0,   # 0 = only prompt at frame 0
) -> Optional[List[np.ndarray]]:
    """
    Runs SAM3 video predictor session over resource_path (mp4 or jpeg-folder).
    Supports multiple prompts by calling add_prompt multiple times.
    Optional anchor_freq: re-add prompts every K frames to reduce drift/flicker.
    Returns list[uint8 mask] length T or None.
    """
    T = len(frames_rgb_u8)
    if T == 0:
        return None
    h, w = frames_rgb_u8[0].shape[:2]

    video_predictor = build_sam3_video_predictor()

    resp = video_predictor.handle_request(request=dict(type="start_session", resource_path=resource_path))
    sid = resp.get("session_id", None)
    if sid is None:
        return None

    # frames at which we re-anchor prompts
    anchor_frames = [0]
    if anchor_freq and anchor_freq > 0:
        anchor_frames += list(range(anchor_freq, T, anchor_freq))

    last_outputs = None
    for fi in anchor_frames:
        for p in prompts:
            resp = video_predictor.handle_request(
                request=dict(type="add_prompt", session_id=sid, frame_index=int(fi), text=p)
            )
            last_outputs = resp.get("outputs", None)

    if last_outputs is None:
        return None

    masks_u8 = _extract_masks_from_video_outputs(last_outputs, T=T, h=h, w=w, score_thresh=score_thresh)
    return masks_u8


# -------------------------
# Main pipeline
# -------------------------
def process_one_pkl(
    pkl_path: str,
    out_root: str,
    prompt: str,
    camera_keys: List[str],
    fps: int,
    overwrite_videos: bool,
    overwrite_pkls: bool,
    use_video_predictor: bool,
    score_thresh: float,
    sam3_segmenter: Optional[Sam3ImageSegmenter] = None,   # NEW
    batch_size=1024,
    video_predictor=None,
    num_chunks=4,
    chunk_overlap=8,
    anchor_freq=32,
):
    transitions = load_pkl(pkl_path)
    if not isinstance(transitions, list):
        raise TypeError(f"{pkl_path} does not contain a list. Got: {type(transitions)}")

    base = os.path.splitext(os.path.basename(pkl_path))[0]

    video_dir = os.path.join(out_root, "videos")
    mask_dir = os.path.join(out_root, "mask_videos")
    overlay_dir = os.path.join(out_root, "overlay_videos")
    pkl_dir = os.path.join(out_root, "pkls")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)

    prompts = parse_prompts(prompt)

    out_transitions = transitions  # in-place OK since output is a new file

    for cam in camera_keys:
        frames = extract_camera_frames(out_transitions, cam, device='cpu')

        # 1) Save original video BEFORE segmentation
        orig_mp4 = os.path.join(video_dir, f"{base}_{cam}_orig.mp4")
        if overwrite_videos or not os.path.exists(orig_mp4):
            write_video(frames, orig_mp4, fps=fps)

        # frames = extract_camera_frames(out_transitions, cam)
                # 2) Segment
        if use_video_predictor:
            tmp_root = os.path.join(out_root, "_tmp_frames", f"{base}_{cam}")

            masks_u8 = segment_video_predictor_num_chunks(
                video_predictor=video_predictor,
                frames_rgb_u8=frames,
                prompts=prompts,
                tmp_root=tmp_root,
                score_thresh=score_thresh,
                num_chunks=num_chunks,
                overlap=chunk_overlap,
                anchor_freq=anchor_freq,          # optional re-anchoring
            )
        else:
            masks_u8 = sam3_segmenter.segment_frames_microbatched(
                frames_rgb_u8=frames,
                prompts=prompts,
                batch_size=int(batch_size),
                score_thresh=score_thresh,
                desc=f"SAM3(image) {base} {cam}",
            )

        # 3) Save mask + overlay videos
        mask_mp4 = os.path.join(mask_dir, f"{base}_{cam}_sam3_mask.mp4")
        overlay_mp4 = os.path.join(overlay_dir, f"{base}_{cam}_sam3_overlay.mp4")
        if overwrite_videos or not os.path.exists(mask_mp4):
            write_mask_video(masks_u8, mask_mp4, fps=fps)
        if overwrite_videos or not os.path.exists(overlay_mp4):
            write_overlay_video(frames, masks_u8, overlay_mp4, fps=fps, alpha=0.55)

        # 4) Insert masks back into pkl structure
        insert_camera_masks(out_transitions, cam, masks_u8, out_key_suffix="_sam3_mask")

    # 5) Save new PKL
    out_pkl = os.path.join(pkl_dir, f"{base}_sam3.pkl")
    print(f"saving to {out_pkl}")
    if overwrite_pkls or not os.path.exists(out_pkl):
        save_pkl(out_transitions, out_pkl)

def process_pkl_dir_batched(
    src_dir: str,
    out_root: str,
    prompt: str,
    camera_keys: List[str],
    fps: int = 30,
    overwrite_videos: bool = False,
    overwrite_pkls: bool = False,
    use_video_predictor: bool = False,
    score_thresh: float = 0.0,
    device: str = "cuda",
    div_num_idx: int = 1,
    div_start_idx: int = 0,
    batch_size=1024,
    num_chunks=4,
    chunk_overlap=8,
    anchor_freq=32,
):
    pkl_paths = list_transition_pkls_batch(src_dir)
    if len(pkl_paths) == 0:
        raise FileNotFoundError(f"No transitions_*.pkl found in {src_dir}")

    # split like your roboengine script
    n = len(pkl_paths)
    per = n // div_num_idx
    start = div_start_idx * per
    end = (div_start_idx + 1) * per
    if div_start_idx == div_num_idx - 1:
        end = n
    pkl_paths = pkl_paths[start:end]

    # build SAM3 ONCE
    sam3_segmenter = Sam3ImageSegmenter(device=device)
    video_predictor = None
    if use_video_predictor:
        video_predictor = build_sam3_video_predictor()

    for pkl_path in tqdm(pkl_paths, desc="PKLs", unit="file"):
        process_one_pkl(
            pkl_path=pkl_path,
            out_root=out_root,
            prompt=prompt,
            camera_keys=camera_keys,
            fps=fps,
            overwrite_videos=overwrite_videos,
            overwrite_pkls=overwrite_pkls,
            use_video_predictor=use_video_predictor,
            score_thresh=score_thresh,
            sam3_segmenter=sam3_segmenter,   # reuse
            batch_size=batch_size,
            video_predictor=video_predictor,
            num_chunks=num_chunks,
            chunk_overlap=chunk_overlap,
            anchor_freq=anchor_freq,
        )

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--input", default="/home/agiuser/experiments" # red block|block stand with hole|red block in block stand|block stand filled by red block
    #                                             "/hil_serl_LOGS/pretraining/" \
    #                                             "camera_pos_metacond_awac_pretrain_randomZ/" \
    #                                             "pos1/online_buffer", help="A transitions_*.pkl file OR a directory containing transitions_*.pkl files.")
    # ap.add_argument("--input", default="/home/agiuser/experiments/hil_serl_LOGS/" \
    #                                     "pretraining/camera_pos_metacond_sac_pretrain_randomZ"
    #                                     "/pos0/online_buffer"
    #                                     , help="A transitions_*.pkl file OR a directory containing transitions_*.pkl files.")
    ap.add_argument("--input", default="/home/agiuser/experiments/hil_serl_LOGS/" \
    "                                   pretraining/yangyang_data/" \
    "                                   precision_connector_demo_buffer"
                                        , help="A transitions_*.pkl file OR a directory containing transitions_*.pkl files.")
    
    ap.add_argument("--commit", default="seg", help="Suffix for output dir name: <input>_sam3_<commit>")
    ap.add_argument("--prompt", required=True, help='Text prompt for SAM3. Use "|" to union prompts, e.g. "peg|hole".')
    ap.add_argument("--camera_keys", default="wrist_1,wrist_2", help="Comma-separated camera keys to segment.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overwrite_videos", default=True,action="store_true")
    ap.add_argument("--overwrite_pkls", default=True,action="store_true")
    ap.add_argument("--use_video_predictor", action="store_true",
                    help="Try SAM3 video_predictor first (single prompt only). Falls back to image-per-frame if needed.")
    ap.add_argument("--score_thresh", type=float, default=0.0,
                    help="If SAM3 returns multiple masks with scores, union only masks with score >= this.")
    ap.add_argument("--start_idx", type=int, default=0, help="Shard start index (like your roboengine script).")
    ap.add_argument("--div_num_idx", type=int, default=1, help="Shard count (like your roboengine script).")
    ap.add_argument("--device", type=str, default="cuda", help="which compute device to use")
    ap.add_argument("--batch_size", type=str, default=1024, help="segmnentationb batch size")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_overlap", type=int, default=16)
    ap.add_argument("--anchor_freq", type=int, default=0)
    

    args = ap.parse_args()

    src_dir = os.path.abspath(args.input.rstrip(os.sep))
    out_root = f"{args.input.rstrip(os.sep)}_sam3_{args.commit}" if os.path.isdir(args.input) else \
               os.path.join(os.path.dirname(args.input), f"_sam3_{args.commit}")
    os.makedirs(out_root, exist_ok=True)

    camera_keys = [c.strip() for c in args.camera_keys.split(",") if c.strip()]

    process_pkl_dir_batched(
        src_dir=src_dir,
        out_root=out_root,
        prompt=args.prompt,
        camera_keys=camera_keys,
        fps=args.fps,
        overwrite_videos=args.overwrite_videos,
        overwrite_pkls=args.overwrite_pkls,
        use_video_predictor=args.use_video_predictor,
        score_thresh=args.score_thresh,
        device=args.device,
        div_num_idx=args.div_num_idx,
        div_start_idx=args.start_idx,
        batch_size=args.batch_size,
        num_chunks=args.num_chunks,
        chunk_overlap=args.chunk_overlap,
        anchor_freq=args.anchor_freq,
    )


if __name__ == "__main__":
    main()
