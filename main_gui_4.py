# main_gui.py
# Accuracy-first Parallel YOLOv8 + MiDaS GUI
# - letterbox preprocessing
# - robust ONNX output handling (1,C,N) and (1,N,C)
# - NMS, confidence slider, IoU slider
# - model selector (use yolov8s.onnx for best accuracy)
# - keeps MiDaS separate
#
# Run: python3 main_gui.py

import os, sys, time, threading, traceback
from glob import glob

import cv2
import numpy as np
import psutil
import onnxruntime as ort
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

# ------------ Config ------------
# Default MiDaS path (unchanged)
MIDAS_PATH = "Midas-V2.onnx"

# Confidence defaults
CONF_THRESHOLD = 0.40
NMS_IOU_THRESHOLD = 0.45

# COCO class labels (80)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair dryer","toothbrush"
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ------------ Helpers ------------
def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    """
    Resize+pad an image to new_shape while keeping aspect ratio.
    Returns padded_img (uint8), (ratio, pad_w_left, pad_h_top)
    """
    h0, w0 = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w0, new_h / h0)
    unp_w, unp_h = int(round(w0 * r)), int(round(h0 * r))
    dw = new_w - unp_w
    dh = new_h - unp_h
    dw_left = dw // 2
    dh_top = dh // 2
    resized = cv2.resize(img, (unp_w, unp_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    padded[dh_top:dh_top+unp_h, dw_left:dw_left+unp_w] = resized
    return padded, (r, dw_left, dh_top)


def xywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2]-box1[0]) * max(0.0, box1[3]-box1[1])
    area2 = max(0.0, box2[2]-box2[0]) * max(0.0, box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size:
        i = int(idxs[0]); keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        rem = []
        for j in rest:
            if iou(boxes[i], boxes[int(j)]) <= iou_thresh:
                rem.append(j)
        idxs = np.array(rem, dtype=int)
    return keep


# ------------ GUI App ------------
class ParallelInferenceGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Springer Research: Edge Parallel Inference (Accuracy mode)")
        self.geometry("1150x820")

        # runtime state
        self.current_image = None  # RGB numpy
        self.processing = False
        self.img_refs = {}

        # ONNX sessions + meta
        self.yolo_sess = None
        self.midas_sess = None
        self.yolo_input_name = None
        self.midas_input_name = None
        self.yolo_W = 640; self.yolo_H = 640
        self.midas_W = 384; self.midas_H = 384
        self._yolo_letterbox = (1.0, 0, 0)

        # timings
        self.last_yolo_ms = None; self.last_midas_ms = None

        # build UI
        self.setup_gui()
        self.find_models_and_set_default()
        self.init_models()   # loads chosen YOLO and MiDaS if present
        self.update_system_stats()

    def setup_gui(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)

        # Left sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="SYSTEM MONITOR", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=16)

        self.core_frame = ctk.CTkFrame(self.sidebar); self.core_frame.pack(pady=8, padx=12, fill="x")
        self.core_labels = []
        for i in range(4):
            lbl = ctk.CTkLabel(self.core_frame, text=f"Core {i}: --%", font=ctk.CTkFont(size=12)); lbl.pack(); self.core_labels.append(lbl)

        self.load_btn = ctk.CTkButton(self.sidebar, text="📁 Load Image", command=self.load_image); self.load_btn.pack(pady=8, padx=12, fill="x")
        self.run_btn = ctk.CTkButton(self.sidebar, text="🚀 Run Parallel Inference", fg_color="#28a745", hover_color="#218838", command=self.start_inference_thread); self.run_btn.pack(pady=8, padx=12, fill="x")

        # model selector
        ctk.CTkLabel(self.sidebar, text="YOLO model (choose for accuracy)").pack(pady=(8,2), padx=12)
        self.model_var = ctk.StringVar(value="")
        self.model_dropdown = ctk.CTkComboBox(self.sidebar, values=[], variable=self.model_var, command=self.on_model_change)
        self.model_dropdown.pack(padx=12, pady=(0,8), fill="x")

        # confidence slider
        ctk.CTkLabel(self.sidebar, text="Confidence threshold").pack(pady=(6,0), padx=12)
        self.conf_slider = ctk.CTkSlider(self.sidebar, from_=0.01, to=0.9, number_of_steps=89, command=self._on_conf_change)
        self.conf_slider.set(CONF_THRESHOLD); self.conf_slider.pack(padx=12, pady=(2,6), fill="x")
        self.conf_label = ctk.CTkLabel(self.sidebar, text=f"Threshold: {CONF_THRESHOLD:.2f}"); self.conf_label.pack(padx=12, pady=(0,8))

        # NMS slider
        ctk.CTkLabel(self.sidebar, text="NMS IoU threshold").pack(pady=(6,0), padx=12)
        self.nms_slider = ctk.CTkSlider(self.sidebar, from_=0.1, to=0.9, number_of_steps=80, command=self._on_nms_change)
        self.nms_slider.set(NMS_IOU_THRESHOLD); self.nms_slider.pack(padx=12, pady=(2,6), fill="x")
        self.nms_label = ctk.CTkLabel(self.sidebar, text=f"IoU: {NMS_IOU_THRESHOLD:.2f}"); self.nms_label.pack(padx=12, pady=(0,8))

        # Metrics panel
        self.metrics_frame = ctk.CTkFrame(self.sidebar); self.metrics_frame.pack(pady=8, padx=12, fill="both", expand=True)
        ctk.CTkLabel(self.metrics_frame, text="METRICS", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.lbl_yolo = ctk.CTkLabel(self.metrics_frame, text="YOLO: -- ms"); self.lbl_yolo.pack()
        self.lbl_midas = ctk.CTkLabel(self.metrics_frame, text="MiDaS: -- ms"); self.lbl_midas.pack()
        self.lbl_total = ctk.CTkLabel(self.metrics_frame, text="Total: -- ms", font=ctk.CTkFont(weight="bold")); self.lbl_total.pack(pady=6)
        self.lbl_count = ctk.CTkLabel(self.metrics_frame, text="Count: --"); self.lbl_count.pack(pady=(6,0))

        # Right content grid
        self.tabs = ctk.CTkTabview(self); self.tabs.grid(row=0, column=1, padx=14, pady=14, sticky="nsew"); self.tabs.add("Visual Results")
        self.grid_frame = ctk.CTkFrame(self.tabs.tab("Visual Results")); self.grid_frame.pack(fill="both", expand=True)
        self.res_labels = {}
        titles = [("Original",0,0), ("Detections",0,1), ("Depth Map",1,0), ("Combined",1,1)]
        for title, r, c in titles:
            f = ctk.CTkFrame(self.grid_frame); f.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            ctk.CTkLabel(f, text=title).pack()
            lbl = ctk.CTkLabel(f, text="No Image Loaded"); lbl.pack(expand=True, fill="both")
            self.res_labels[title] = lbl
            self.grid_frame.grid_columnconfigure(c, weight=1); self.grid_frame.grid_rowconfigure(r, weight=1)

    def _on_conf_change(self, val):
        global CONF_THRESHOLD
        CONF_THRESHOLD = float(val)
        self.conf_label.configure(text=f"Threshold: {CONF_THRESHOLD:.2f}")

    def _on_nms_change(self, val):
        global NMS_IOU_THRESHOLD
        NMS_IOU_THRESHOLD = float(val)
        self.nms_label.configure(text=f"IoU: {NMS_IOU_THRESHOLD:.2f}")

    # discover ONNX models present and populate dropdown; prefer bigger models for accuracy
    def find_models_and_set_default(self):
        files = sorted(glob("yolov8*.onnx"))
        if not files:
            # also accept any .onnx file with "yolo" substring
            files = sorted([f for f in glob("*.onnx") if "yolo" in f.lower()])
        if not files:
            # fallback to expected default name
            files = ["yolov8n.onnx"]  # may not exist; init_models will warn
        self.model_dropdown.configure(values=files)
        # pick best available by heuristic: prefer "s" > "m" > "n"
        preferred = None
        for pref in ("yolov8s.onnx","yolov8m.onnx","yolov8n.onnx"):
            if pref in files:
                preferred = pref; break
        if not preferred:
            preferred = files[0]
        self.model_var.set(preferred)

    def on_model_change(self, val):
        # user changed model: reload sessions
        self.init_models()

    # load ONNX sessions
    def init_models(self):
        chosen = self.model_var.get()
        if not chosen:
            chosen = "yolov8n.onnx"
        yolo_path = chosen
        if not os.path.exists(yolo_path):
            messagebox.showwarning("YOLO model missing", f"YOLO file '{yolo_path}' not found in folder.")
            self.run_btn.configure(state="disabled", text="Model missing")
            return
        if not os.path.exists(MIDAS_PATH):
            messagebox.showwarning("MiDaS missing", f"MiDaS file '{MIDAS_PATH}' not found.")
            self.run_btn.configure(state="disabled", text="Model missing")
            return

        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

            self.yolo_sess = ort.InferenceSession(yolo_path, sess_options=so, providers=['CPUExecutionProvider'])
            self.midas_sess = ort.InferenceSession(MIDAS_PATH, sess_options=so, providers=['CPUExecutionProvider'])

            # cache input shapes & names
            y_in = self.yolo_sess.get_inputs()[0]; self.yolo_input_name = y_in.name
            y_shape = [d if isinstance(d, int) else None for d in y_in.shape]
            if len(y_shape) >= 4 and y_shape[2] is not None and y_shape[3] is not None:
                self.yolo_H, self.yolo_W = y_shape[2], y_shape[3]
            else:
                self.yolo_H = self.yolo_W = 640

            m_in = self.midas_sess.get_inputs()[0]; self.midas_input_name = m_in.name
            m_shape = [d if isinstance(d, int) else None for d in m_in.shape]
            if len(m_shape) >= 4 and m_shape[2] is not None and m_shape[3] is not None:
                self.midas_H, self.midas_W = m_shape[2], m_shape[3]
            else:
                self.midas_H = self.midas_W = 384

            print(f"[INFO] Loaded YOLO: {yolo_path} input({self.yolo_H}x{self.yolo_W})  MiDaS: {MIDAS_PATH} input({self.midas_H}x{self.midas_W})")
            self.run_btn.configure(state="normal", text="🚀 Run Parallel Inference")
        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Model load error", f"{e}\n\n{tb}")
            self.run_btn.configure(state="disabled", text="Models missing")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Load error", "Could not read image.")
            return
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_img(self.current_image, "Original")

    def display_img(self, img_rgb, label_key):
        try:
            pil = Image.fromarray(img_rgb)
            img_ctk = ctk.CTkImage(light_image=pil, dark_image=pil, size=(420,320))
            self.res_labels[label_key].configure(image=img_ctk, text="")
            self.img_refs[label_key] = img_ctk
        except Exception as e:
            print("display_img error:", e)
            self.res_labels[label_key].configure(text="Unable to display image")

    def start_inference_thread(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Load an image first!"); return
        if self.processing: return
        if self.yolo_sess is None or self.midas_sess is None:
            messagebox.showwarning("Warning", "Models not loaded"); return
        self.processing = True
        self.run_btn.configure(state="disabled", text="Processing...")
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        start_total = time.time()
        results = {}
        try:
            h_orig, w_orig = self.current_image.shape[:2]

            # YOLO: letterbox resize to model input shape
            yolo_shape = (self.yolo_W, self.yolo_H)
            yolo_padded, (yr, ydw, ydh) = letterbox(self.current_image, new_shape=yolo_shape)
            # we use RGB order (works with standard yolov8 ONNX)
            y_blob = yolo_padded.astype(np.float32).transpose(2,0,1) / 255.0
            y_blob = np.expand_dims(y_blob.astype(np.float32), axis=0)

            # store letterbox metadata for mapping back
            self._yolo_letterbox = (yr, ydw, ydh)

            # MiDaS preprocessing - model-specific
            midas_resized = cv2.resize(self.current_image, (self.midas_W, self.midas_H)).astype(np.float32)
            midas_blob = (midas_resized / 255.0 - 0.5) / 0.5
            midas_blob = midas_blob.transpose(2,0,1).astype(np.float32)
            midas_blob = np.expand_dims(midas_blob, axis=0)

            # run inference in threads
            def run_yolo():
                try:
                    s = time.time(); out = self.yolo_sess.run(None, {self.yolo_input_name: y_blob})
                    results['yolo_raw'] = out; results['yolo_t'] = (time.time() - s) * 1000.0
                except Exception as e:
                    print("YOLO inference error:", e); results['yolo_raw'] = None; results['yolo_t'] = None

            def run_midas():
                try:
                    s = time.time(); out = self.midas_sess.run(None, {self.midas_input_name: midas_blob})
                    results['midas_raw'] = out; results['midas_t'] = (time.time() - s) * 1000.0
                except Exception as e:
                    print("MiDaS inference error:", e); results['midas_raw'] = None; results['midas_t'] = None

            t1 = threading.Thread(target=run_yolo); t2 = threading.Thread(target=run_midas)
            t1.start(); t2.start(); t1.join(); t2.join()

            total_t = (time.time() - start_total) * 1000.0
            self.after(0, lambda: self.finalize_results(results, total_t, w_orig, h_orig))
        except Exception as e:
            tb = traceback.format_exc(); print("run_inference error:", e, tb)
            self.processing = False
            self.after(0, lambda: messagebox.showerror("Inference error", str(e)))
            self.after(0, lambda: self.run_btn.configure(state="normal", text="🚀 Run Parallel Inference"))

    def finalize_results(self, res, total_t, w_orig, h_orig):
        # timings
        if res.get('yolo_t') is not None:
            self.last_yolo_ms = res['yolo_t']; self.lbl_yolo.configure(text=f"YOLO: {self.last_yolo_ms:.1f} ms")
        else:
            self.lbl_yolo.configure(text="YOLO: -- ms")
        if res.get('midas_t') is not None:
            self.last_midas_ms = res['midas_t']; self.lbl_midas.configure(text=f"MiDaS: {self.last_midas_ms:.1f} ms")
        else:
            self.lbl_midas.configure(text="MiDaS: -- ms")
        self.lbl_total.configure(text=f"Total: {total_t:.1f} ms")

        det_img_rgb = self.current_image.copy()
        depth_img_rgb = np.zeros_like(det_img_rgb)

        # ---- Robust YOLO parsing (handles (1,C,N) and (1,N,C)) ----
        boxes, scores, classes = [], [], []
        y_raw = res.get('yolo_raw', None)
        if y_raw is None:
            print("[INFO] YOLO produced no result")
        else:
            arr = y_raw[0] if isinstance(y_raw, (list, tuple)) else y_raw
            arr = np.asarray(arr)
            print("[DEBUG] raw yolo shape:", arr.shape)

            # if (1, C, N) typical from some exporters: transpose to (1,N,C)
            if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] <= 200 and arr.shape[2] > arr.shape[1]:
                arr = arr.transpose(0,2,1)

            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim == 2 and arr.shape[1] >= 5:
                coords = arr[:, :4]; coords_max = float(coords.max()) if coords.size else 0.0
                print(f"[DEBUG] coords_max={coords_max:.4f} yolo_W={self.yolo_W} yolo_H={self.yolo_H}")
                yr, ydw, ydh = self._yolo_letterbox

                for i, row in enumerate(arr):
                    row = row.astype(float); cols = row.shape[0]
                    prob = 0.0; cls_id = None; x1f=y1f=x2f=y2f=None

                    if cols >= 6 and cols > 6:
                        cx,cy,bw,bh = row[:4]; obj_conf = float(row[4]); class_probs = row[5:]
                        if class_probs.size:
                            cls_id = int(np.argmax(class_probs)); cls_prob = float(class_probs[cls_id])
                        else:
                            cls_prob = 1.0
                        final_score = obj_conf * cls_prob
                        final_score = max(final_score, obj_conf)
                        prob = float(final_score)

                        # coords in padded space: if normalized -> multiply by model input dims, else assume model pixels
                        if coords_max <= 1.01:
                            cx_p = cx * self.yolo_W; cy_p = cy * self.yolo_H; bw_p = bw * self.yolo_W; bh_p = bh * self.yolo_H
                        else:
                            cx_p, cy_p, bw_p, bh_p = cx, cy, bw, bh

                        # remove padding then scale back to original image
                        x1_p, y1_p, x2_p, y2_p = xywh_to_xyxy(cx_p, cy_p, bw_p, bh_p)
                        x1 = (x1_p - ydw) / yr; y1 = (y1_p - ydh) / yr
                        x2 = (x2_p - ydw) / yr; y2 = (y2_p - ydh) / yr

                    elif cols == 6:
                        c0,c1,c2,c3 = row[:4]; score = float(row[4]); cls_id = int(row[5]); prob = score
                        if max(c0,c1,c2,c3) <= 1.01:
                            cx_p, cy_p, bw_p, bh_p = c0 * self.yolo_W, c1 * self.yolo_H, c2 * self.yolo_W, c3 * self.yolo_H
                            x1_p,y1_p,x2_p,y2_p = xywh_to_xyxy(cx_p, cy_p, bw_p, bh_p)
                            x1 = (x1_p - ydw)/yr; y1 = (y1_p - ydh)/yr; x2 = (x2_p - ydw)/yr; y2 = (y2_p - ydh)/yr
                        else:
                            # treat as xyxy relative to model input dims or absolute
                            if max(c0,c1,c2,c3) <= max(self.yolo_W, self.yolo_H):
                                sx = w_orig/float(self.yolo_W); sy = h_orig/float(self.yolo_H)
                                x1, y1, x2, y2 = c0*sx, c1*sy, c2*sx, c3*sy
                            else:
                                x1, y1, x2, y2 = c0, c1, c2, c3

                    elif cols == 5:
                        cx,cy,bw,bh = row[:4]; prob = float(row[4])
                        if coords_max <= 1.01:
                            cx_p = cx * self.yolo_W; cy_p = cy * self.yolo_H; bw_p = bw * self.yolo_W; bh_p = bh * self.yolo_H
                        else:
                            cx_p, cy_p, bw_p, bh_p = cx, cy, bw, bh
                        x1_p,y1_p,x2_p,y2_p = xywh_to_xyxy(cx_p, cy_p, bw_p, bh_p)
                        x1 = (x1_p - ydw)/yr; y1 = (y1_p - ydh)/yr; x2 = (x2_p - ydw)/yr; y2 = (y2_p - ydh)/yr
                    else:
                        continue

                    # int coords
                    try:
                        x1i = int(max(0, round(x1))); y1i = int(max(0, round(y1)))
                        x2i = int(min(w_orig - 1, round(x2))); y2i = int(min(h_orig - 1, round(y2)))
                    except Exception:
                        continue

                    if i < 6:
                        print(f"[YOLO CAND] idx={i} cls={cls_id} prob={prob:.4f} box={(x1i,y1i,x2i,y2i)} cols={cols}")

                    if prob >= CONF_THRESHOLD:
                        boxes.append([x1i,y1i,x2i,y2i]); scores.append(float(prob)); classes.append(int(cls_id) if cls_id is not None else -1)
            else:
                print("[DEBUG] unexpected YOLO layout:", arr.shape)

        # draw NMS results (on BGR then convert)
        det_img_bgr = cv2.cvtColor(det_img_rgb, cv2.COLOR_RGB2BGR)
        keep = nms(boxes, scores, NMS_IOU_THRESHOLD) if boxes else []
        detected_count = len(keep)
        for k in keep:
            x1,y1,x2,y2 = boxes[k]; score = scores[k]; cls_id = classes[k]
            color_bgr = (240,32,160)
            cv2.rectangle(det_img_bgr, (x1,y1), (x2,y2), color_bgr, 2)
            cls_name = COCO_CLASSES[cls_id] if (0 <= cls_id < len(COCO_CLASSES)) else str(cls_id)
            label = f"{cls_name} {score:.2f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lbx1 = x1; lby1 = max(0, y1-th-6); lbx2 = x1+tw+6; lby2 = y1
            cv2.rectangle(det_img_bgr, (lbx1,lby1), (lbx2,lby2), color_bgr, -1)
            cv2.putText(det_img_bgr, label, (lbx1+3, lby2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        count_label = f"Detected: {detected_count}"
        (cw,ch),_ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(det_img_bgr, (8,8), (8+cw+12, 8+ch+10), (0,0,0), -1)
        cv2.putText(det_img_bgr, count_label, (12,8+ch+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        self.lbl_count.configure(text=f"Count: {detected_count}")
        det_img_rgb = cv2.cvtColor(det_img_bgr, cv2.COLOR_BGR2RGB)

        # -------- MiDaS postprocessing ----------
        m_raw = res.get('midas_raw', None)
        if m_raw is None:
            print("[INFO] MiDaS produced no result")
        else:
            m = m_raw[0] if isinstance(m_raw, (list,tuple)) else m_raw
            m = np.asarray(m)
            if m.ndim == 4 and m.shape[0] == 1: m = m[0]
            if m.ndim == 3 and m.shape[0] == 1: m = m[0]
            if m.ndim == 2:
                depth_norm = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
                depth_img_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                depth_img_rgb = cv2.resize(depth_img_rgb, (w_orig, h_orig))
            else:
                print("[DEBUG] MiDaS unexpected shape:", m.shape)

        # combined overlay
        try:
            combined = cv2.addWeighted(det_img_rgb, 0.6, depth_img_rgb, 0.4, 0.0)
        except Exception:
            combined = det_img_rgb.copy()

        # update UI
        self.display_img(det_img_rgb, "Detections")
        self.display_img(depth_img_rgb, "Depth Map")
        self.display_img(combined, "Combined")

        # finish
        self.processing = False
        if self.yolo_sess and self.midas_sess: self.run_btn.configure(state="normal", text="🚀 Run Parallel Inference")
        else: self.run_btn.configure(state="disabled", text="Models missing")

    def update_system_stats(self):
        cpu_usage = psutil.cpu_percent(percpu=True)
        for i, usage in enumerate(cpu_usage):
            if i < 4: self.core_labels[i].configure(text=f"Core {i}: {usage:.1f}%")
        self.after(1000, self.update_system_stats)


if __name__ == "__main__":
    app = ParallelInferenceGUI()
    app.mainloop()
