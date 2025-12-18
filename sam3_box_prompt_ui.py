"""
SAM3 visual prompt UI (positive / negative exemplar prompts).

Now supports:
- BOX tool (existing): left-mouse drag => {"xywh":[x,y,w,h], "label":bool}
- LASSO tool (new):  left-mouse drag freehand => {"poly":[[x,y],...], "label":bool}

Controls
  - Draw: left mouse drag
  - p / n : switch draw label to Positive / Negative
  - m     : toggle tool BOX <-> LASSO
  - a / d : previous / next frame (within provided frame_indices list)
  - [ / ] : previous / next prompt
  - u     : undo last prompt item for current (prompt, frame)
  - c     : clear all prompt items for current (prompt, frame)
  - h     : print help to terminal
  - x     : print summary + confirm, then save & exit
  - q or ESC : quit without saving

Return format (JSON-serializable)
    dict[prompt][frame_idx] -> list[
        {"kind":"box","xywh":[x,y,w,h],"label":bool} OR
        {"kind":"poly","poly":[[x,y],...],"label":bool}
    ]

Notes
- Everything is stored in ORIGINAL pixel coordinates.
- Display is auto-scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _AnnItem:
    kind: str                 # "box" or "poly"
    label: bool               # True=POS (green), False=NEG (red)
    xywh: Optional[List[float]] = None
    poly: Optional[List[List[float]]] = None  # [[x,y], ...] in ORIGINAL coords


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
        lasso_simplify_epsilon_px: float = 2.0,  # simplification in ORIGINAL pixels
        ui_tool: str = "box"
    ):
        if not frames_rgb_u8:
            raise ValueError("frames_rgb_u8 is empty")
        if not prompts:
            raise ValueError("prompts is empty")
        if not frame_indices:
            raise ValueError("frame_indices is empty")

        self.frames = frames_rgb_u8
        self.prompts = list(prompts)

        self.frame_indices, self._clamp_report = self._sanitize_frame_indices(
            frame_indices, n_frames=len(frames_rgb_u8)
        )

        self.window_name = window_name
        self.status_window_name = status_window_name
        self.max_display_wh = tuple(max_display_wh)
        self.scale_up_max = float(scale_up_max)
        self.show_status_window = bool(show_status_window)
        self.lasso_simplify_epsilon_px = float(lasso_simplify_epsilon_px)
        

        # annotations[prompt][frame_idx] = list[_AnnItem]
        self.annotations: Dict[str, Dict[int, List[_AnnItem]]] = {
            p: {int(fi): [] for fi in self.frame_indices} for p in self.prompts
        }

        # UI state
        self._frame_pos = 0
        self._prompt_pos = 0
        self._cur_label = True  # True=POS, False=NEG
        self._tool = "box"      # "box" or "lasso"
        ui_tool = (ui_tool or "box").lower()
        if ui_tool not in ("box", "lasso", "both"):
            ui_tool = "box"

        self._ui_tool_mode = ui_tool
        self._allow_toggle = (ui_tool == "both")
        self._tool = "box" if ui_tool in ("box", "both") else "lasso"

        # dragging state
        self._dragging = False
        self._p0: Optional[Tuple[int, int]] = None   # display-space start (box)
        self._p1: Optional[Tuple[int, int]] = None   # display-space current (box)

        # lasso state (display-space points)
        self._lasso_pts: List[Tuple[int, int]] = []

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

            # save & exit
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

            # tool toggle
            # tool toggle
            if key == ord("m"):
                if not self._allow_toggle:
                    self._notify(f"Tool toggle disabled (ui_tool='{self._ui_tool_mode}').")
                    continue

                self._tool = "lasso" if self._tool == "box" else "box"
                # cancel any partial gesture
                self._dragging = False
                self._p0 = self._p1 = None
                self._lasso_pts = []
                self._notify(self._status_line(prefix=f"Tool changed: {self._tool.upper()}"))
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
                items = self._cur_items()
                if items:
                    removed = items.pop()
                    if removed.kind == "box":
                        self._notify(
                            f"Undo: removed {'POS' if removed.label else 'NEG'} BOX xywh={self._fmt_xywh(removed.xywh)}. Remaining={len(items)}"
                        )
                    else:
                        self._notify(
                            f"Undo: removed {'POS' if removed.label else 'NEG'} LASSO poly_pts={len(removed.poly or [])}. Remaining={len(items)}"
                        )
                else:
                    self._notify("Undo: no items to remove.")
                continue

            if key == ord("c"):
                n = len(self._cur_items())
                self._set_cur_items([])
                self._notify(f"Cleared {n} items for current (prompt, frame).")
                continue

            self._notify(self._status_line(prefix=f"Ignored keycode={key}"), also_print=False)

    # -----------------
    # Rendering / status
    # -----------------
    def _render_image_only(self) -> np.ndarray:
        fr = self._cur_frame()
        h, w = fr.shape[:2]
        s = float(self._scale)

        # RGB -> BGR
        bgr = fr[:, :, ::-1].copy()

        if s != 1.0:
            interp = cv2.INTER_NEAREST if s > 1.0 else cv2.INTER_AREA
            bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=interp)

        # draw saved items
        for it in self._cur_items():
            color = (0, 255, 0) if it.label else (0, 0, 255)
            if it.kind == "box" and it.xywh is not None:
                x, y, bw, bh = it.xywh
                x0, y0 = int(round(x * s)), int(round(y * s))
                x1, y1 = int(round((x + bw) * s)), int(round((y + bh) * s))
                cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
            elif it.kind == "poly" and it.poly:
                pts = np.asarray(it.poly, dtype=np.float32) * s
                pts = pts.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(bgr, [pts], isClosed=True, color=color, thickness=2)

        # draw in-progress gesture
        if self._tool == "box":
            if self._dragging and self._p0 is not None and self._p1 is not None:
                x0, y0 = self._p0
                x1, y1 = self._p1
                color = (0, 255, 0) if self._cur_label else (0, 0, 255)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        else:  # lasso
            if self._dragging and len(self._lasso_pts) >= 2:
                color = (0, 255, 0) if self._cur_label else (0, 0, 255)
                pts = np.array(self._lasso_pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(bgr, [pts], isClosed=False, color=color, thickness=2)

        return bgr

    def _render_status_window(self) -> np.ndarray:
        W, H = 1040, 280
        img = np.full((H, W, 3), 255, dtype=np.uint8)

        lines = [
            self._status_line(prefix="Current"),
            f"Last: {self._last_msg}",
            ("Keys: p/n label | m tool | a/d frame | [/]=prompt | u undo | c clear | x save+confirm | q quit"
                if self._allow_toggle else
                "Keys: p/n label | a/d frame | [/]=prompt | u undo | c clear | x save+confirm | q quit"),
            f"Tool: {self._tool.upper()} | Display scale: {self._scale:.3f} (stored in ORIGINAL coords)",
        ]

        y = 28
        for line in lines:
            for chunk in self._wrap_text(line, width=135):
                cv2.putText(img, chunk, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
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
        x = int(x); y = int(y)

        if self._tool == "box":
            self._on_mouse_box(event, x, y)
        else:
            self._on_mouse_lasso(event, x, y)

    def _on_mouse_box(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._p0 = (x, y)
            self._p1 = (x, y)
            self._notify(self._status_line(prefix="Drag start (BOX)"))
            return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._p1 = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self._p1 = (x, y)
            if self._p0 is None or self._p1 is None:
                return

            x0, y0 = self._p0
            x1, y1 = self._p1
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])

            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                self._p0, self._p1 = None, None
                self._notify("Ignored tiny drag (too small).")
                return

            s = float(self._scale)
            x_min_f = float(x_min) / s
            y_min_f = float(y_min) / s
            w_f = float(x_max - x_min) / s
            h_f = float(y_max - y_min) / s

            fr = self._cur_frame()
            H0, W0 = fr.shape[:2]
            x_min_f = float(np.clip(x_min_f, 0.0, max(0.0, W0 - 1.0)))
            y_min_f = float(np.clip(y_min_f, 0.0, max(0.0, H0 - 1.0)))
            w_f = float(np.clip(w_f, 1.0, max(1.0, W0 - x_min_f)))
            h_f = float(np.clip(h_f, 1.0, max(1.0, H0 - y_min_f)))

            it = _AnnItem(kind="box", xywh=[x_min_f, y_min_f, w_f, h_f], label=bool(self._cur_label))
            self._cur_items().append(it)
            self._p0, self._p1 = None, None

            self._notify(f"Added {'POS' if it.label else 'NEG'} BOX xywh={self._fmt_xywh(it.xywh)}")
            return

    def _on_mouse_lasso(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._lasso_pts = [(x, y)]
            self._notify(self._status_line(prefix="Drag start (LASSO)"))
            return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            # append point if moved enough (avoid huge point lists)
            if not self._lasso_pts:
                self._lasso_pts = [(x, y)]
                return
            lx, ly = self._lasso_pts[-1]
            if (x - lx) * (x - lx) + (y - ly) * (y - ly) >= 4:  # >=2px
                self._lasso_pts.append((x, y))
            return

        if event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            if len(self._lasso_pts) < 6:
                self._lasso_pts = []
                self._notify("Ignored lasso: too few points.")
                return

            # display -> original
            s = float(self._scale)
            pts = np.asarray(self._lasso_pts, dtype=np.float32)
            pts[:, 0] /= s
            pts[:, 1] /= s

            fr = self._cur_frame()
            H0, W0 = fr.shape[:2]
            pts[:, 0] = np.clip(pts[:, 0], 0.0, float(W0 - 1))
            pts[:, 1] = np.clip(pts[:, 1], 0.0, float(H0 - 1))

            # simplify (in original pixels)
            cnt = pts.reshape(-1, 1, 2)
            eps = max(0.5, self.lasso_simplify_epsilon_px)
            approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
            poly = approx.reshape(-1, 2)

            if poly.shape[0] < 3:
                self._lasso_pts = []
                self._notify("Ignored lasso: degenerate polygon.")
                return

            area = float(cv2.contourArea(poly.astype(np.float32)))
            if area < 10.0:
                self._lasso_pts = []
                self._notify("Ignored lasso: area too small.")
                return

            poly_list = [[float(x), float(y)] for x, y in poly.tolist()]

            it = _AnnItem(kind="poly", poly=poly_list, label=bool(self._cur_label))
            self._cur_items().append(it)
            self._lasso_pts = []

            self._notify(f"Added {'POS' if it.label else 'NEG'} LASSO poly_pts={len(poly_list)} area={area:.1f}")
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
            "Draw BOX:   tool=BOX, left mouse drag\n"
            "Draw LASSO: tool=LASSO, left mouse drag around region\n"
            "p/n: Positive/Negative mode\n"
            + ("" if not self._allow_toggle else "m: toggle tool BOX<->LASSO\n")
            + "a/d: prev/next frame (in your ui_frames list)\n"
            "[/]: prev/next prompt\n"
            "u: undo last item\n"
            "c: clear items for current (prompt, frame)\n"
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

        # de-dupe preserve order
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
        mw, mh = self.max_display_wh
        fit = min(float(mw) / float(w), float(mh) / float(h))
        if fit > 1.0:
            return float(min(fit, self.scale_up_max))
        return float(max(fit, 1e-6))

    def _on_frame_changed(self):
        h, w = self._cur_frame().shape[:2]
        self._scale = self._compute_scale(w, h)
        # reset in-progress
        self._dragging = False
        self._p0 = self._p1 = None
        self._lasso_pts = []
        self._notify(self._status_line(prefix="Frame changed"))

    def _status_line(self, prefix: str = "Status") -> str:
        fi = self._cur_frame_idx()
        pidx = self._prompt_pos
        prompt = self._cur_prompt()
        mode = "POS" if self._cur_label else "NEG"
        tool = self._tool.upper()
        items = self._cur_items()
        pos_n = sum(1 for it in items if it.label)
        neg_n = len(items) - pos_n
        return (
            f"{prefix} | frame={fi} ({self._frame_pos+1}/{len(self.frame_indices)}) | "
            f"prompt[{pidx}]='{prompt}' ({self._prompt_pos+1}/{len(self.prompts)}) | "
            f"tool={tool} | mode={mode} | items: POS={pos_n}, NEG={neg_n}"
        )

    def _cur_prompt(self) -> str:
        return self.prompts[self._prompt_pos]

    def _cur_frame_idx(self) -> int:
        return int(self.frame_indices[self._frame_pos])

    def _cur_frame(self) -> np.ndarray:
        return self.frames[self._cur_frame_idx()]

    def _cur_items(self) -> List[_AnnItem]:
        return self.annotations[self._cur_prompt()][self._cur_frame_idx()]

    def _set_cur_items(self, items: List[_AnnItem]):
        self.annotations[self._cur_prompt()][self._cur_frame_idx()] = items

    def _export(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for p, by_f in self.annotations.items():
            out[p] = {}
            for fi in self.frame_indices:
                items = by_f.get(int(fi), [])
                row: List[Dict[str, Any]] = []
                for it in items:
                    if it.kind == "box" and it.xywh is not None:
                        row.append({"kind": "box", "xywh": it.xywh, "label": bool(it.label)})
                    elif it.kind == "poly" and it.poly:
                        row.append({"kind": "poly", "poly": it.poly, "label": bool(it.label)})
                out[p][int(fi)] = row
        return out

    def _print_summary(self, export: Dict[str, Dict[int, List[Dict[str, Any]]]]):
        print("\n=== SAM3 Visual Prompt Summary ===")
        for p in self.prompts:
            print(f"\nPrompt: '{p}'")
            by_f = export.get(p, {})
            for fi in self.frame_indices:
                items = by_f.get(int(fi), [])
                if not items:
                    print(f"  frame {fi}: (no items)")
                    continue
                print(f"  frame {fi}:")
                for it in items:
                    kind = it.get("kind", "?")
                    lab = "POS" if bool(it.get("label", False)) else "NEG"
                    if kind == "box":
                        print(f"    {lab} BOX {self._fmt_xywh(it.get('xywh', [0,0,0,0]))}")
                    elif kind == "poly":
                        pts = it.get("poly", [])
                        print(f"    {lab} LASSO poly_pts={len(pts)}")
                    else:
                        print(f"    {lab} {kind}")
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
