import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from defurnishing.geometry.cubemap import (
    CUBE_FACE_KEYS,
    equirectangular_to_cubemap,
    cubemap_to_equirectangular,
)

def main():
    img_path = Path("outputs/Havana.jpeg")
    out_dir = Path("outputs")
    
    if not img_path.exists():
        print(f"Error: {img_path} not found!")
        return

    print(f"Loading {img_path}...")
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")

    face_size = 1024
    print(f"1. Splitting into 6 Cubemap faces (face_size={face_size}x{face_size})...")
    t0 = time.time()
    faces = equirectangular_to_cubemap(img, face_size=face_size)
    t_split = time.time() - t0
    print(f"   Done in {t_split:.3f}s")

    # Save all 6 cube faces
    face_names = {
        "F": "Front",
        "R": "Right",
        "B": "Back",
        "L": "Left",
        "U": "Up (Ceiling)",
        "D": "Down (Floor)"
    }
    for k in CUBE_FACE_KEYS:
        out_face_path = out_dir / f"cube_{k}.jpg"
        cv2.imwrite(str(out_face_path), faces[k], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   Saved {face_names[k]}: {out_face_path}")

    print(f"2. Merging 6 faces back to Equirectangular ({w}x{h})...")
    t0 = time.time()
    reconstructed = cubemap_to_equirectangular(faces, out_width=w, out_height=h)
    t_merge = time.time() - t0
    print(f"   Done in {t_merge:.3f}s")

    reconstructed_path = out_dir / "Havana_reconstructed.jpg"
    cv2.imwrite(str(reconstructed_path), reconstructed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"   Saved reconstructed panorama: {reconstructed_path}")

    # Compute reconstruction difference (excluding pole singularity pixels if any)
    diff = np.abs(img.astype(np.float32) - reconstructed.astype(np.float32))
    mae = np.mean(diff)
    psnr = cv2.PSNR(img, reconstructed)
    print(f"3. Quality metrics:")
    print(f"   MAE (Mean Absolute Error): {mae:.2f} / 255")
    print(f"   PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    main()

