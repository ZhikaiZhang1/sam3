"""SAM3 visual prompt UI (positive / negative exemplar boxes).

This UI is meant for *image-per-frame* prompting:
- You provide a list of frames and a list of text prompts.
- For each (prompt, frame_idx) pair, you can draw multiple POS / NEG exemplar boxes.

Important behaviors
- The display is auto-scaled up/down for comfortable drawing.
- Boxes are stored/exported in ORIGINAL pixel coordinates.
- No text is drawn on the image (use the separate status window + terminal).
- Every operation prints a confirmation message.
- Positive boxes are GREEN, negative boxes are RED.
- On exit ('x'), the UI prints a full summary and asks you to confirm.

Return format (JSON-serializable)
    dict[prompt][frame_idx] -> list[{"xywh": [x,y,w,h], "label": bool}]

Controls
  - Draw: left mouse drag
  - p / n : switch draw label to Positive / Negative
  - a / d : previous / next frame (within provided frame_indices list)
  - [ / ] : previous / next prompt
  - u     : undo last box for current (prompt, frame)
  - c     : clear all boxes for current (prompt, frame)
  - h     : print help to terminal
  - x     : print summary + confirm, then save & exit
  - q or ESC : quit without saving
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Box:
    xywh: List[float]  # [x,y,w,h] in ORIGINAL pixel coords
    label: bool        # True=POS (green), False=NEG (red)


class Sam3BoxPromptUI:
    def __init__(
        self,
        frames_rgb_u8: List[np.ndarray],
        prompts: List[str],
        frame_indices: List[int],
        window_name: str = "SAM3 Visual Prompts",
        status_window_name: str = "SAM3 Status",
        max_display_wh: Tuple[int, int] = (1400, 900),
        scale_up_max: float = 8.0,
        show_status_window: bool = True,
    ):
        if not frames_rgb_u8:
            raise ValueError("frames_rgb_u8 is empty")
        if not prompts:
            raise ValueError("prompts is empty")
        if not frame_indices:
            raise ValueError("frame_indices is empty")

        self.frames = frames_rgb_u8
        self.prompts = list(prompts)

        # Clamp indices into [0, len(frames)-1]; keep order.
        self.frame_indices, self._clamp_report = self._sanitize_frame_indices(
            frame_indices, n_frames=len(frames_rgb_u8)
        )

        self.window_name = window_name
        self.status_window_name = status_window_name
        self.max_display_wh = tuple(max_display_wh)
        self.scale_up_max = float(scale_up_max)
        self.show_status_window = bool(show_status_window)

        # annotations[prompt][frame_idx] = list[_Box]
        self.annotations: Dict[str, Dict[int, List[_Box]]] = {
            p: {int(fi): [] for fi in self.frame_indices} for p in self.prompts
        }

        # UI state
        self._frame_pos = 0
        self._prompt_pos = 0
        self._cur_label = True  # True=POS, False=NEG
        self._dragging = False
        self._p0: Optional[Tuple[int, int]] = None  # display-space start
        self._p1: Optional[Tuple[int, int]] = None  # display-space current

        self._last_msg = "Ready. Press 'h' for help."

        # scaling (updated when frame changes)
        h, w = self._cur_frame().shape[:2]
        self._scale = self._compute_scale(w, h)

        # windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        if self.show_status_window:
            cv2.namedWindow(self.status_window_name, cv2.WINDOW_NORMAL)

    # -----------------
    # Public
    # -----------------
    def run(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """Blocking UI loop."""
        self._print_help()
        if self._clamp_report:
            self._notify(self._clamp_report)

        self._notify(self._status_line(prefix="Ready"), also_print=False)

        while True:
            cv2.imshow(self.window_name, self._render_image_only())
            if self.show_status_window:
                cv2.imshow(self.status_window_name, self._render_status_window())

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # quit without saving
            if key in (ord("q"), 27):  # ESC
                self._notify("Quit without saving.")
                self._destroy_windows()
                raise KeyboardInterrupt("Visual prompt UI aborted")

            # help
            if key == ord("h"):
                self._print_help()
                self._notify("Help printed.")
                continue

            # save & exit (with summary + confirm)
            if key == ord("x"):
                export = self._export()
                self._print_summary(export)
                if self._confirm("Proceed with these visual prompts? [y/N]: "):
                    self._notify("Confirmed. Saving & exiting.")
                    self._destroy_windows()
                    return export
                self._notify("Not confirmed. Continue editing.")
                continue

            # label mode
            if key == ord("p"):
                self._cur_label = True
                self._notify(self._status_line(prefix="Label mode: POSITIVE"))
                continue
            if key == ord("n"):
                self._cur_label = False
                self._notify(self._status_line(prefix="Label mode: NEGATIVE"))
                continue

            # navigation
            if key == ord("a"):
                old = self._frame_pos
                self._frame_pos = max(0, self._frame_pos - 1)
                if self._frame_pos != old:
                    self._on_frame_changed()
                else:
                    self._notify(self._status_line(prefix="Frame unchanged (already first)"))
                continue

            if key == ord("d"):
                old = self._frame_pos
                self._frame_pos = min(len(self.frame_indices) - 1, self._frame_pos + 1)
                if self._frame_pos != old:
                    self._on_frame_changed()
                else:
                    self._notify(self._status_line(prefix="Frame unchanged (already last)"))
                continue

            if key == ord("["):
                old = self._prompt_pos
                self._prompt_pos = max(0, self._prompt_pos - 1)
                if self._prompt_pos != old:
                    self._notify(self._status_line(prefix="Prompt changed"))
                else:
                    self._notify(self._status_line(prefix="Prompt unchanged (already first)"))
                continue

            if key == ord("]"):
                old = self._prompt_pos
                self._prompt_pos = min(len(self.prompts) - 1, self._prompt_pos + 1)
                if self._prompt_pos != old:
                    self._notify(self._status_line(prefix="Prompt changed"))
                else:
                    self._notify(self._status_line(prefix="Prompt unchanged (already last)"))
                continue

            # edits
            if key == ord("u"):
                boxes = self._cur_boxes()
                if boxes:
                    removed = boxes.pop()
                    self._notify(
                        f"Undo: removed {'POS' if removed.label else 'NEG'} box xywh={self._fmt_xywh(removed.xywh)}. Remaining={len(boxes)}"
                    )
                else:
                    self._notify("Undo: no boxes to remove.")
                continue

            if key == ord("c"):
                n = len(self._cur_boxes())
                self._set_cur_boxes([])
                self._notify(f"Cleared {n} boxes for current (prompt, frame).")
                continue

            # unknown key: ignore
            self._notify(self._status_line(prefix=f"Ignored keycode={key}"), also_print=False)

    # -----------------
    # Rendering / status
    # -----------------
    def _render_image_only(self) -> np.ndarray:
        """Return only the image with rectangles; NO text overlays."""
        fr = self._cur_frame()
        h, w = fr.shape[:2]
        s = float(self._scale)

        # RGB -> BGR for OpenCV display
        bgr = fr[:, :, ::-1].copy()

        if s != 1.0:
            interp = cv2.INTER_NEAREST if s > 1.0 else cv2.INTER_AREA
            bgr = cv2.resize(
                bgr,
                (int(round(w * s)), int(round(h * s))),
                interpolation=interp,
            )

        # draw saved boxes
        for b in self._cur_boxes():
            x, y, bw, bh = b.xywh
            x0, y0 = int(round(x * s)), int(round(y * s))
            x1, y1 = int(round((x + bw) * s)), int(round((y + bh) * s))
            color = (0, 255, 0) if b.label else (0, 0, 255)  # POS green, NEG red
            cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)

        # draw in-progress drag rectangle
        if self._dragging and self._p0 is not None and self._p1 is not None:
            x0, y0 = self._p0
            x1, y1 = self._p1
            color = (0, 255, 0) if self._cur_label else (0, 0, 255)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)

        return bgr

    def _render_status_window(self) -> np.ndarray:
        """Separate status window (NOT overlaid on the frame)."""
        W, H = 980, 260
        img = np.full((H, W, 3), 255, dtype=np.uint8)

        lines = [
            self._status_line(prefix="Current"),
            f"Last: {self._last_msg}",
            "Keys: p/n label | a/d frame | [/]=prompt | u undo | c clear | x save+confirm | q quit",
            f"Display scale: {self._scale:.3f} (boxes stored in ORIGINAL coords)",
        ]

        y = 28
        for line in lines:
            for chunk in self._wrap_text(line, width=125):
                cv2.putText(
                    img,
                    chunk,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                y += 24
                if y > H - 12:
                    break
            if y > H - 12:
                break

        return img

    # -----------------
    # Mouse handling
    # -----------------
    def _on_mouse(self, event, x, y, flags, param):
        # x,y are in DISPLAY-space (after scaling)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._p0 = (int(x), int(y))
            self._p1 = (int(x), int(y))
            self._notify(self._status_line(prefix="Drag start"))
            return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._p1 = (int(x), int(y))
            return

        if event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self._p1 = (int(x), int(y))
            if self._p0 is None or self._p1 is None:
                return

            x0, y0 = self._p0
            x1, y1 = self._p1
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])

            # ignore tiny drags
            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                self._p0, self._p1 = None, None
                self._notify("Ignored tiny drag (too small).")
                return

            # display-space -> original pixel-space
            s = float(self._scale)
            x_min_f = float(x_min) / s
            y_min_f = float(y_min) / s
            w_f = float(x_max - x_min) / s
            h_f = float(y_max - y_min) / s

            # clamp into image bounds (original)
            fr = self._cur_frame()
            H0, W0 = fr.shape[:2]
            x_min_f = float(np.clip(x_min_f, 0.0, max(0.0, W0 - 1.0)))
            y_min_f = float(np.clip(y_min_f, 0.0, max(0.0, H0 - 1.0)))
            w_f = float(np.clip(w_f, 1.0, max(1.0, W0 - x_min_f)))
            h_f = float(np.clip(h_f, 1.0, max(1.0, H0 - y_min_f)))

            box = _Box(xywh=[x_min_f, y_min_f, w_f, h_f], label=bool(self._cur_label))
            self._cur_boxes().append(box)

            self._p0, self._p1 = None, None

            self._notify(f"Added {'POS' if box.label else 'NEG'} box xywh={self._fmt_xywh(box.xywh)}")
            return

    # -----------------
    # Helpers
    # -----------------
    def _destroy_windows(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        if self.show_status_window:
            try:
                cv2.destroyWindow(self.status_window_name)
            except Exception:
                pass

    def _notify(self, msg: str, also_print: bool = True):
        self._last_msg = msg
        if also_print:
            print(msg)

    def _print_help(self):
        print(
            "\n[SAM3 Visual Prompts UI]\n"
            "Draw box: left mouse drag\n"
            "p/n: Positive/Negative mode\n"
            "a/d: prev/next frame (in your ui_frames list)\n"
            "[/]: prev/next prompt\n"
            "u: undo last box\n"
            "c: clear boxes for current (prompt, frame)\n"
            "x: print summary + confirm, then save & exit\n"
            "q or ESC: quit without saving\n"
        )

    def _confirm(self, prompt: str) -> bool:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            ans = ""
        return ans in ("y", "yes")

    @staticmethod
    def _sanitize_frame_indices(frame_indices: List[int], n_frames: int) -> Tuple[List[int], str]:
        """Clamp indices to a valid range.

        Requested behavior: if a frame index is larger than the file length, clamp it to the last frame.
        Also clamps negatives to 0.

        Returns (sanitized_indices, report_string).
        """
        if n_frames <= 0:
            raise ValueError("n_frames must be > 0")

        raw_list: List[int] = []
        for raw in frame_indices:
            try:
                raw_list.append(int(raw))
            except Exception:
                continue

        if not raw_list:
            return [n_frames - 1], f"[ui_frames] empty/invalid -> using last frame {n_frames - 1}"

        out: List[int] = []
        changed: List[Tuple[int, int]] = []
        for i in raw_list:
            if i < 0:
                i2 = 0
            elif i >= n_frames:
                i2 = n_frames - 1
            else:
                i2 = i
            out.append(i2)
            if i2 != i:
                changed.append((i, i2))

        # de-dupe while preserving order (after clamping)
        seen = set()
        uniq: List[int] = []
        for i in out:
            if i not in seen:
                uniq.append(i)
                seen.add(i)

        if changed:
            report = "; ".join([f"{a}->{b}" for a, b in changed])
            return uniq, f"[ui_frames] clamped out-of-range indices: {report}"

        return uniq, ""

    def _compute_scale(self, w: int, h: int) -> float:
        """Scale so image fits into max_display_wh.

        - If frame is huge: scale down to fit.
        - If frame is tiny: scale up to fit, capped by scale_up_max.
        """
        mw, mh = self.max_display_wh
        fit = min(float(mw) / float(w), float(mh) / float(h))
        if fit > 1.0:
            return float(min(fit, self.scale_up_max))
        return float(max(fit, 1e-6))

    def _on_frame_changed(self):
        h, w = self._cur_frame().shape[:2]
        self._scale = self._compute_scale(w, h)
        self._notify(self._status_line(prefix="Frame changed"))

    def _status_line(self, prefix: str = "Status") -> str:
        fi = self._cur_frame_idx()
        pidx = self._prompt_pos
        prompt = self._cur_prompt()
        mode = "POS" if self._cur_label else "NEG"
        boxes = self._cur_boxes()
        pos_n = sum(1 for b in boxes if b.label)
        neg_n = len(boxes) - pos_n
        return (
            f"{prefix} | frame={fi} ({self._frame_pos+1}/{len(self.frame_indices)}) | "
            f"prompt[{pidx}]='{prompt}' ({self._prompt_pos+1}/{len(self.prompts)}) | "
            f"mode={mode} | boxes: POS={pos_n}, NEG={neg_n}"
        )

    def _cur_prompt(self) -> str:
        return self.prompts[self._prompt_pos]

    def _cur_frame_idx(self) -> int:
        return int(self.frame_indices[self._frame_pos])

    def _cur_frame(self) -> np.ndarray:
        return self.frames[self._cur_frame_idx()]

    def _cur_boxes(self) -> List[_Box]:
        return self.annotations[self._cur_prompt()][self._cur_frame_idx()]

    def _set_cur_boxes(self, boxes: List[_Box]):
        self.annotations[self._cur_prompt()][self._cur_frame_idx()] = boxes

    def _export(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for p, by_f in self.annotations.items():
            out[p] = {}
            for fi in self.frame_indices:
                boxes = by_f.get(int(fi), [])
                out[p][int(fi)] = [{"xywh": b.xywh, "label": bool(b.label)} for b in boxes]
        return out

    def _print_summary(self, export: Dict[str, Dict[int, List[Dict[str, Any]]]]):
        print("\n=== SAM3 Visual Prompt Summary ===")
        for p in self.prompts:
            print(f"\nPrompt: '{p}'")
            by_f = export.get(p, {})
            for fi in self.frame_indices:
                items = by_f.get(int(fi), [])
                pos = [it["xywh"] for it in items if bool(it.get("label", False))]
                neg = [it["xywh"] for it in items if not bool(it.get("label", False))]
                if not pos and not neg:
                    print(f"  frame {fi}: (no boxes)")
                else:
                    print(f"  frame {fi}:")
                    if pos:
                        print("    POS:")
                        for b in pos:
                            print(f"      {self._fmt_xywh(b)}")
                    if neg:
                        print("    NEG:")
                        for b in neg:
                            print(f"      {self._fmt_xywh(b)}")
        print("=== End Summary ===\n")

    @staticmethod
    def _wrap_text(s: str, width: int = 120) -> List[str]:
        words = s.split(" ")
        out: List[str] = []
        cur: List[str] = []
        n = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if n + add > width:
                out.append(" ".join(cur))
                cur = [w]
                n = len(w)
            else:
                if cur:
                    cur.append(w)
                    n += add
                else:
                    cur = [w]
                    n = len(w)
        if cur:
            out.append(" ".join(cur))
        return out

    @staticmethod
    def _fmt_xywh(xywh: List[float]) -> str:
        x, y, w, h = xywh
        return f"[x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]"