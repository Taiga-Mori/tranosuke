import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pathlib import Path
import threading
import time
import os

from PIL import Image, ImageTk

import streamlit as st
from pathlib import Path
import time
from typing import List
import numpy as np
import pandas as pd
import os
import pydomino
import librosa
import soundfile as sf

from utils import *
from utterance import *
from morpheme import *
from phoneme import *
from word import *

# --- 画像パス ---
IMG1_PATH = "asset/tranosuke_square.png"
IMG2_PATH = "asset/tranosuke_transcribing.png"
IMG3_PATH = "asset/tranosuke_done.png"

# --- Tkinterアプリ ---
class TranscribeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("書き起こしアプリ とらのすけ")
        self.geometry("600x500")
        
        # --- 状態 ---
        self.state = "start"  # start / processing / done
        
        # --- 画像 ---
        self.img_label = tk.Label(self)
        self.img_label.pack(pady=10)
        
        self.load_image(IMG1_PATH)
        
        # --- 入力フォーム ---
        self.audio_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.quality_var = tk.StringVar(value="スピード優先")
        
        self.start_frame = tk.Frame(self)
        self.start_frame.pack(pady=10)
        
        tk.Label(self.start_frame, text="音声ファイル:").grid(row=0, column=0, sticky="e")
        tk.Entry(self.start_frame, textvariable=self.audio_path_var, width=40).grid(row=0, column=1)
        tk.Button(self.start_frame, text="参照", command=self.browse_audio).grid(row=0, column=2)
        
        tk.Label(self.start_frame, text="出力ディレクトリ:").grid(row=1, column=0, sticky="e")
        tk.Entry(self.start_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1)
        tk.Button(self.start_frame, text="参照", command=self.browse_output).grid(row=1, column=2)
        
        tk.Label(self.start_frame, text="品質:").grid(row=2, column=0, sticky="e")
        tk.Radiobutton(self.start_frame, text="スピード優先", variable=self.quality_var, value="スピード優先").grid(row=2, column=1, sticky="w")
        tk.Radiobutton(self.start_frame, text="クオリティ優先", variable=self.quality_var, value="クオリティ優先").grid(row=2, column=1)
        
        tk.Button(self.start_frame, text="書き起こし実行！", command=self.start_processing).grid(row=3, column=1, pady=10)
        
        # --- 処理中の進捗表示 ---
        self.progress_label = tk.Label(self, text="", fg="blue")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=400, mode="indeterminate")
        
        # --- 完了画面 ---
        self.done_frame = tk.Frame(self)
        self.done_label = tk.Label(self.done_frame, text="", fg="green")
        self.done_label.pack()
        tk.Button(self.done_frame, text="次のファイルを書き起こす", command=self.reset).pack(pady=10)
        
    def load_image(self, path):
        img = Image.open(path)
        img = img.resize((200,200))
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.photo)
        
    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("音声ファイル", "*.mp3 *.wav")])
        if path:
            self.audio_path_var.set(path)
    
    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)
    
    def start_processing(self):
        audio_path = Path(self.audio_path_var.get())
        output_dir = Path(self.output_dir_var.get())
        quality = self.quality_var.get()
        
        if not audio_path.exists():
            messagebox.showerror("エラー", "音声ファイルが見つかりません")
            return
        
        self.state = "processing"
        self.start_frame.pack_forget()
        self.load_image(IMG2_PATH)
        self.progress_label.config(text="処理中...")
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
        
        # --- 処理を別スレッドで実行 ---
        threading.Thread(target=self.run_processing, args=(audio_path, output_dir, quality)).start()
    
    def run_processing(self, audio_path, output_dir, quality):
        # --- 処理内容をここに置き換え ---
        # 例として時間待機で擬似処理
        time.sleep(2)  # 辞書とモデル準備
        self.progress_label.config(text="書き起こし中...")
        time.sleep(2)  # 書き起こし
        self.progress_label.config(text="形態素解析中...")
        time.sleep(2)  # 形態素解析
        self.progress_label.config(text="強制アラインメント中...")
        time.sleep(2)  # 強制アラインメント
        self.progress_label.config(text="単語情報更新中...")
        time.sleep(2)  # 単語処理
        
        # --- 保存 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # df_phon.to_csv(output_dir / "phoneme.csv")
        # df_word.to_csv(output_dir / "word.csv")
        # df_utt_adj.to_csv(output_dir / "utterance.csv")
        
        # --- 完了画面に切り替え ---
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        self.load_image(IMG3_PATH)
        self.done_label.config(text=f"終わったよ！ 結果を {output_dir} に保存したよ🐯")
        self.done_frame.pack(pady=20)
    
    def reset(self):
        self.done_frame.pack_forget()
        self.start_frame.pack(pady=10)
        self.load_image(IMG1_PATH)
        self.audio_path_var.set("")
        self.output_dir_var.set("")
        self.quality_var.set("スピード優先")
        self.state = "start"

if __name__ == "__main__":
    app = TranscribeApp()
    app.mainloop()
