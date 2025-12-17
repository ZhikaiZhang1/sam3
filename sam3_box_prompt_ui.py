"""SAM3 visual prompt UI (positive / negative exemplar boxes).

This module is intentionally standalone so you can keep your main segmentation script
from becoming even longer.

Usage (from your script):

    ui = Sam3BoxPromptUI(
        frames_rgb_u8=frames,
        prompts=prompts,
        frame_indices=[0, 10, 20],
    )
    ann = ui.run()  # dict[prompt][frame_idx] -> list[{xywh:[x,y,w,h], label:bool}]

Keys (in the UI window):
  - Draw: left mouse drag
  - p / n : switch draw label to Positive / Negative
  - a / d : previous / next frame (within the provided frame_indices list)
  - [ / ] : previous / next prompt
  - u     : undo last box for current (prompt+ frame)
  - c     : clear all boxes for current (prompt+ frame)
  - h     : print help to terminal
  - x     : save & exit (returns annotations)
  - q or ESC : quit without saving (raises KeyboardInterrupt)

Notes:
  - Positive boxes are GREEN; Negative boxes are RED.
  - Every key action prints a confirmation message to the terminal and updates an on-screen status line.
  - When you press 'x', the UI prints a full per-(prompt, frame) summary and asks you to confirm before exiting.

Example Sequence:
  Pick a prompt (use [ / ] until you're on prompt[i]='objX').

  Pick a frame (use a / d until you're on the frame you want).

  Press p (POS mode), then draw all positive boxes for that prompt on that frame.

  Press n (NEG mode), then draw all negative boxes (common confusers, background junk, merged regions you want excluded).

  Move to next frame (d) and repeat steps 3-4 for the same prompt.

  When done with that prompt across your selected frames, go to next prompt (]) and repeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _Box:
    xywh: List[float]  # [x,y,w,h] in original pixel coords
    label: bool        # True=positive (green), False=negative (red)


class Sam3BoxPromptUI:
    def __init__(
        self,
        frames_rgb_u8: List[np.ndarray],
        prompts: List[str],
        frame_indices: List[int],
        window_name: str = "SAM3 Visual Prompts",
        max_display_wh: Tuple[int, int] = (1400, 900),
    ):
        if not frames_rgb_u8:
            raise ValueError("frames_rgb_u8 is empty")
        if not prompts:
            raise ValueError("prompts is empty")
        if not frame_indices:
            raise ValueError("frame_indices is empty")

        self.frames = frames_rgb_u8
        self.prompts = prompts
        self.frame_indices = frame_indices
        self.window_name = window_name
        self.max_display_wh = max_display_wh

        # annotations[prompt][frame_idx] = list[_Box]
        self.annotations: Dict[str, Dict[int, List[_Box]]] = {
            p: {int(fi): [] for fi in frame_indices} for p in prompts
        }

        # UI state
        self._frame_pos = 0
        self._prompt_pos = 0
        self._cur_label = True  # True=pos, False=neg
        self._dragging = False
        self._p0: Optional[Tuple[int, int]] = None  # display-space start
        self._p1: Optional[Tuple[int, int]] = None  # display-space current

        # scaling
        h, w = self._cur_frame().shape[:2]
        self._scale = self._compute_scale(w, h)

        # window init
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    # -----------------
    # Public API
    # -----------------
    def run(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """Block until user saves or quits.

        Returns JSON-serializable annotations:
          dict[prompt][frame_idx] -> list[{"xywh": [x,y,w,h], "label": bool}]
        """
        self._last_msg = "Ready. Press 'h' for help."
        self._print_help()
        self._notify(self._status_line(), also_print=False)

        while True:
            disp = self._render()
            cv2.imshow(self.window_name, disp)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # quit without saving
            if key in (ord("q"), 27):  # ESC
                cv2.destroyWindow(self.window_name)
                raise KeyboardInterrupt("Visual prompt UI aborted")

            # save & exit
            if key == ord("x"):
                # Print summary + require confirmation BEFORE returning.
                export = self._export()
                self._print_summary(export)
                if self._confirm("Proceed with these visual prompts? [y/N]: "):
                    cv2.destroyWindow(self.window_name)
                    self._notify("Confirmed. Saving & exiting.")
                    return export
                else:
                    self._notify("Not confirmed. Continue editing.")
                    continue

            if key == ord("h"):
                self._print_help()
                self._notify("Help printed to terminal.")

            # label mode
            if key == ord("p"):
                self._cur_label = True
                self._notify(self._status_line(prefix="Label mode: POSITIVE"))
            if key == ord("n"):
                self._cur_label = False
                self._notify(self._status_line(prefix="Label mode: NEGATIVE"))

            # navigation
            if key == ord("a"):
                old = self._frame_pos
                self._frame_pos = max(0, self._frame_pos - 1)
                if self._frame_pos != old:
                    self._on_frame_changed()
                else:
                    self._notify(self._status_line(prefix="Frame unchanged (already first)"))
            if key == ord("d"):
                old = self._frame_pos
                self._frame_pos = min(len(self.frame_indices) - 1, self._frame_pos + 1)
                if self._frame_pos != old:
                    self._on_frame_changed()
                else:
                    self._notify(self._status_line(prefix="Frame unchanged (already last)"))
            if key == ord("["):
                old = self._prompt_pos
                self._prompt_pos = max(0, self._prompt_pos - 1)
                if self._prompt_pos != old:
                    self._on_prompt_changed()
                else:
                    self._notify(self._status_line(prefix="Prompt unchanged (already first)"))
            if key == ord("]"):
                old = self._prompt_pos
                self._prompt_pos = min(len(self.prompts) - 1, self._prompt_pos + 1)
                if self._prompt_pos != old:
                    self._on_prompt_changed()
                else:
                    self._notify(self._status_line(prefix="Prompt unchanged (already last)"))

            # edit
            if key == ord("u"):
                boxes = self._cur_boxes()
                if boxes:
                    boxes.pop()
                    self._notify(f"Undo: removed last box. Remaining={len(boxes)}")
                else:
                    self._notify("Undo: no boxes to remove.")
            if key == ord("c"):
                self._set_cur_boxes([])
                self._notify("Cleared boxes for current (prompt, frame).")

    # -----------------
    # Internals
    # -----------------
    def _print_help(self):
        print(
            "\n[SAM3 Visual Prompts UI]\n"
            "Draw: left mouse drag\n"
            "p/n: Positive/Negative mode\n"
            "a/d: prev/next frame (in your --ui_frames list)\n"
            "[/]: prev/next prompt\n"
            "u: undo last box\n"
            "c: clear boxes for current (prompt, frame)\n"
            "x: save & exit\n"
            "q or ESC: quit without saving\n"
        )

    def _notify(self, msg: str, also_print: bool = True):
        """Record a status message (shown on-screen) and optionally print it."""
        self._last_msg = msg
        if also_print:
            print(f"[UI] {msg}")

    def _confirm(self, prompt: str) -> bool:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

    def _status_line(self, prefix: str = "Status") -> str:
        p = self._cur_prompt()
        fi = self._cur_frame_idx()
        mode = "POS" if self._cur_label else "NEG"
        return (
            f"{prefix}: frame={fi} ({self._frame_pos+1}/{len(self.frame_indices)}), "
            f"prompt[{self._prompt_pos}]='{p}', mode={mode}, boxes={len(self._cur_boxes())}"
        )

    def _print_summary(self, export: Dict[str, Dict[int, List[Dict[str, Any]]]]):
        """Print per-(prompt, frame) boxes in a human-readable way."""
        print("\n================ VISUAL PROMPTS SUMMARY ================")
        for p in self.prompts:
            print(f"\nPROMPT[{self.prompts.index(p)}]: {p}")
            by_f = export.get(p, {})
            for fi in self.frame_indices:
                boxes = by_f.get(int(fi), [])
                pos = [b["xywh"] for b in boxes if b.get("label", True)]
                neg = [b["xywh"] for b in boxes if not b.get("label", True)]
                if not boxes:
                    print(f"  frame {int(fi)}: (no boxes)")
                    continue
                def _fmt(xywh):
                    x, y, w, h = xywh
                    xi, yi, wi, hi = int(round(x)), int(round(y)), int(round(w)), int(round(h))
                    return f"[{xi},{yi},{wi},{hi}] (float: {x:.1f},{y:.1f},{w:.1f},{h:.1f})"
                pos_s = ", ".join(_fmt(x) for x in pos) if pos else "(none)"
                neg_s = ", ".join(_fmt(x) for x in neg) if neg else "(none)"
                print(f"  frame {int(fi)}: POS={pos_s} | NEG={neg_s}")
        print("\n========================================================\n")

    def _on_frame_changed(self):
        """Update any frame-dependent state when user navigates frames."""
        fr = self._cur_frame()
        h, w = fr.shape[:2]
        self._scale = self._compute_scale(w, h)
        self._notify(self._status_line(prefix="Frame changed"))

    def _on_prompt_changed(self):
        """Update any prompt-dependent state when user navigates prompts."""
        self._notify(self._status_line(prefix="Prompt changed"))

    def _compute_scale(self, w: int, h: int) -> float:
        mw, mh = self.max_display_wh
        s = min(float(mw) / float(w), float(mh) / float(h), 1.0)
        return s

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
            for fi, boxes in by_f.items():
                out[p][int(fi)] = [{"xywh": b.xywh, "label": bool(b.label)} for b in boxes]
        return out

    def _render(self) -> np.ndarray:
        fr = self._cur_frame()
        h, w = fr.shape[:2]
        s = self._scale

        # RGB->BGR for display
        bgr = fr[:, :, ::-1].copy()
        if s != 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        # draw saved boxes
        for bi, b in enumerate(self._cur_boxes()):
            x, y, bw, bh = b.xywh
            x0, y0 = int(round(x * s)), int(round(y * s))
            x1, y1 = int(round((x + bw) * s)), int(round((y + bh) * s))
            color = (0, 255, 0) if b.label else (0, 0, 255)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
            tag = f"{'P' if b.label else 'N'}{bi}"
            cv2.putText(bgr, tag, (max(0, x0 + 3), max(15, y0 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # draw in-progress drag rect
        if self._dragging and self._p0 is not None and self._p1 is not None:
            x0, y0 = self._p0
            x1, y1 = self._p1
            color = (0, 255, 0) if self._cur_label else (0, 0, 255)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)

        # HUD
        prompt = self._cur_prompt()
        fi = self._cur_frame_idx()
        mode = "POS" if self._cur_label else "NEG"
        hud = f"frame={fi} ({self._frame_pos+1}/{len(self.frame_indices)})  prompt={prompt} ({self._prompt_pos+1}/{len(self.prompts)})  mode={mode}  boxes={len(self._cur_boxes())}"
        cv2.putText(bgr, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, "Drag LMB to add box. p/n label. a/d frame. [/]=prompt. x save.", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"{self._last_msg}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return bgr

    def _on_mouse(self, event, x, y, flags, param):
        # Note: x,y are in display-space coordinates (after scaling).
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._p0 = (int(x), int(y))
            self._p1 = (int(x), int(y))
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
            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                # ignore tiny accidental clicks
                return

            # convert display-space -> original pixel-space
            s = self._scale
            x_min_f = float(x_min) / s
            y_min_f = float(y_min) / s
            w_f = float(x_max - x_min) / s
            h_f = float(y_max - y_min) / s

            self._cur_boxes().append(_Box(xywh=[x_min_f, y_min_f, w_f, h_f], label=bool(self._cur_label)))
            lab = "POS" if self._cur_label else "NEG"
            self._notify(
                f"Added {lab} box on frame={self._cur_frame_idx()} prompt='{self._cur_prompt()}': "
                f"xywh=[{int(round(x_min_f))},{int(round(y_min_f))},{int(round(w_f))},{int(round(h_f))}] (px)"
            )
            self._p0, self._p1 = None, None
