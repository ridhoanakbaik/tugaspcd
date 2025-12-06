import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Metode Segmentasi Citra")

        self.original_image = None
        self.result_image = None

        # Frame kontrol
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(control_frame, text="Metode:").pack(side=tk.LEFT)

        self.method_var = tk.StringVar(value="Edge Detection (Canny)")
        methods = [
            "Edge Detection (Canny)",
            "Thresholding (Otsu)",
            "Region Growing",
            "Split and Merge",
            "Clustering (K-Means)",
        ]
        self.method_combo = ttk.Combobox(
            control_frame, textvariable=self.method_var,
            values=methods, state="readonly", width=25
        )
        self.method_combo.pack(side=tk.LEFT, padx=5)

        btn_apply = tk.Button(control_frame, text="Apply", command=self.apply_segmentation)
        btn_apply.pack(side=tk.LEFT, padx=10)

        btn_reset = tk.Button(control_frame, text="Reset", command=self.reset_image)
        btn_reset.pack(side=tk.LEFT)

        # ====== NEW tombol save ======
        btn_save = tk.Button(
            control_frame,
            text="Save Result",
            command=self.save_result
        )
        btn_save.pack(side=tk.LEFT, padx=10)
        # ===============================

        # Frame gambar
        image_frame = tk.Frame(root, padx=10, pady=10)
        image_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.LabelFrame(image_frame, text="Citra Asli")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.original_label = tk.Label(left_frame, bg="grey")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        right_frame = tk.LabelFrame(image_frame, text="Hasil Segmentasi")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.result_label = tk.Label(right_frame, bg="grey")
        self.result_label.pack(fill=tk.BOTH, expand=True)

    # =====================================================
    # SIMPAN HASIL
    # =====================================================
    def save_result(self):  # NEW
        if self.result_image is None:
            messagebox.showwarning("Peringatan", "Tidak ada hasil untuk disimpan.")
            return

        filetypes = [
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("BMP", "*.bmp"),
            ("All Files", "*.*"),
        ]

        filename = filedialog.asksaveasfilename(
            title="Simpan Hasil",
            defaultextension=".png",
            filetypes=filetypes
        )

        if not filename:
            return

        cv2.imwrite(filename, self.result_image)

        messagebox.showinfo("Berhasil", f"Hasil disimpan di:\n{filename}")

    # =====================================================

    def load_image(self):
        filetypes = [
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Pilih Citra", filetypes=filetypes
        )
        if not filename:
            return

        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", "Gagal membaca file citra.")
            return

        self.original_image = img
        self.result_image = None
        self.show_image(self.original_image, self.original_label)
        self.result_label.configure(image='')
        self.result_label.image = None

    def reset_image(self):
        if self.original_image is None:
            return
        self.result_image = None
        self.show_image(self.original_image, self.original_label)
        self.result_label.configure(image='')
        self.result_label.image = None

    def show_image(self, image, label_widget):
        if image is None:
            return

        if len(image.shape) == 2:
            image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_display = image.copy()

        h, w = image_display.shape[:2]
        max_w, max_h = 400, 400
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return

        image_display = cv2.resize(image_display, (new_w, new_h))
        image_rgb = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        label_widget.configure(image=imgtk)
        label_widget.image = imgtk

    # ========= FUNGSI SEGMENTASI ========= #

    def apply_segmentation(self):
        if self.original_image is None:
            messagebox.showwarning("Peringatan", "Silakan load citra terlebih dahulu.")
            return

        method = self.method_var.get()

        try:
            if method == "Edge Detection (Canny)":
                result = self.edge_detection(self.original_image)
            elif method == "Thresholding (Otsu)":
                result = self.thresholding_otsu(self.original_image)
            elif method == "Region Growing":
                result = self.region_growing(self.original_image)
            elif method == "Split and Merge":
                result = self.split_and_merge(self.original_image)
            elif method == "Clustering (K-Means)":
                result = self.kmeans_clustering(self.original_image)
            else:
                messagebox.showerror("Error", "Metode tidak dikenali.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat segmentasi:\n{e}")
            return

        self.result_image = result
        self.show_image(self.result_image, self.result_label)

    # --- Metode Segmentasi ---

    def edge_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def thresholding_otsu(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def region_growing(self, img, tolerance=5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        seed_y, seed_x = h//2, w//2
        seed_value = int(gray[seed_y, seed_x])

        mask = np.zeros_like(gray, np.uint8)
        visited = np.zeros_like(gray, np.bool_)

        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True

        while stack:
            y, x = stack.pop()
            mask[y, x] = 255

            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny,nx]:
                        if abs(int(gray[ny,nx]) - seed_value) <= tolerance:
                            stack.append((ny,nx))
                        visited[ny,nx] = True

        return mask

    def split_and_merge(self, img, std_thresh=10, min_size=32):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        result = np.zeros_like(gray, np.uint8)

        def process(y,x,hh,ww):
            region = gray[y:y+hh, x:x+ww]
            if hh <= min_size or ww <= min_size or np.std(region) < std_thresh:
                mean = int(np.mean(region))
                result[y:y+hh, x:x+ww] = mean
            else:
                hh2, ww2 = hh//2, ww//2
                if hh2==0 or ww2==0:
                    result[y:y+hh, x:x+ww] = int(np.mean(region))
                    return
                process(y,x,hh2,ww2)
                process(y,x+ww2,hh2,ww-ww2)
                process(y+hh2,x,hh-hh2,ww2)
                process(y+hh2,x+ww2,hh-hh2,ww-ww2)

        process(0,0,h,w)
        return result

    def kmeans_clustering(self, img, k=3):
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        result = res.reshape(img.shape)
        return result


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationGUI(root)
    root.mainloop()
