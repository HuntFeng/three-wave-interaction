import numpy as np
import os
import shutil

def clear_datadir():
    if os.path.exists("data"):
        shutil.rmtree("data")
    
    os.mkdir("data")
    os.mkdir("data/a0")
    os.mkdir("data/a1")
    os.mkdir("data/a2")

def save_data(frame: int, a, to_numpy=False):
    for mu in range(3):
        data = a[mu].to_numpy() if to_numpy else a[mu]
        np.save(f"data/a{mu}/{frame}", data)