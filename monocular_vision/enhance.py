## Image enhancement using decomposition for images captured in low-light conditions

"""
Notes from the paper:

Image decomposition model:
- Decompose the night image (I) into the structure layer (Is) and the texture layer (It).
                I = Is + It
- Structure layer calculations:
    - Smooth renderings of the input image - increase the brightness of the structure layer to avoid noise amplification.
    - Obtaining the structure layer - Rolling guidance filter (RGF from now on) (yet another edge retaining filer) to smooth the night image.
    - RGF: Small structure removal + edge recovery. (Pg 5 eq 2 & 3)
    -           Is = RGF(I, σs, σr, t)
    -           It = I = Is
- Brightness enhancement of structural layer:
    - Enhancement processing of RGB - color distortion : choose HSV (Hue, Saturation, Value/Luminance) instead represented by (Ih, Is, Iv).
    - Linear guidance filter with smooth and bordering functions to estimate the light component and uses brightness component Iv as input images and guide images.
    - Luminance components are processed by two guidance filters and fused together by averaging and weighting as the final component estimate. 
- Stretch the structural layer saturation
- Denoise the texture layer using the BM3D denoising algorithm.
- Enhance edges - effective guide filtering (EGIF)

Final Algorithm:
Step 1: Enter a low-illumination image I.
Step 2: Obtain the structure layer Is using Formula (6).
Step 3: Obtain the texture layer It using Formula (7).
Step 4: Enhance the brightness of the structural layer using Formula (11).
Step 5: Stretch the structural layer saturation using Formula (12).
Step 6: Denoise the texture layer using Formula (13).
Step 7: Obtain the fused image R(x, y) using Formula (14).
Step 8: Enhance edge to fused image using Formula (36).
Step 9: Output enhanced image output.

"""

"""
Corrected implementation for:
'Nighttime Image Stitching Method Based on Image Decomposition Enhancement'
- Keep everything in float32 [0,1] internally.
- Use BM3D if available, otherwise use OpenCV NLMeans fallback.
- Stable edge enhancement with clamped beta.
- Saves intermediate images for inspection.
"""

"""
Final corrected enhancement pipeline
- Default RGF params chosen from your sweep: sigma_s=2.5, sigma_r=0.2, iter=3
- Internal computations in float32 [0,1]
- Uses bm3d package if available (preferred); otherwise uses OpenCV NLMeans as fallback.
- Saves intermediate images to out_final/
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from bm3d import bm3d


IMAGE_PATH = "../data/nighttime4.jpg"  
OUT_DIR = "./out_final"
os.makedirs(OUT_DIR, exist_ok=True)

# params (as close to the ones given in the paper)
RGF_SIGMA_S = 2.5
RGF_SIGMA_R = 0.2  
RGF_ITER = 3
R1, LAMBDA1 = 15, 0.1**2
R2, LAMBDA2 = 15, 0.01**2
GAMMA_TEXTURE = 2.2
EDGE_RADIUS = 15
EDGE_EPS = 0.01**2
GAMMA_EDGE = 0.9
BETA_CLAMP_MAX = 10.0
EPS = 1e-6

def save(img_float, name):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, np.clip(img_float * 255.0, 0, 255).astype(np.uint8))
    return path

def stats(tag, arr):
    print(f"{tag}: min {arr.min():.6f}, max {arr.max():.6f}, mean {arr.mean():.6f}, std {arr.std():.6f}")

# Load orginal image 
img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
img = img_bgr.astype(np.float32) / 255.0
stats("original", img)
save(img, "00_original.png")


# Get structure layer using RGF
def compute_rgf(img_float, sigma_s, sigma_r, num_iter):
    # Try float RGF (sigmaColor in same scale as img [0,1])
    try:
        struct = cv2.ximgproc.rollingGuidanceFilter(
            img_float.astype(np.float32), d=-1,
            sigmaColor=float(sigma_r), sigmaSpace=float(sigma_s), numOfIter=int(num_iter)
        ).astype(np.float32)
        return struct, "float"
    except Exception:
        # Fallback: scale to uint8 and call RGF with sigmaColor scaled to 0..255
        img_u8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
        sigmaColor_u8 = max(1, int(sigma_r * 255.0))
        struct_u8 = cv2.ximgproc.rollingGuidanceFilter(
            img_u8, d=-1, sigmaColor=sigmaColor_u8, sigmaSpace=float(sigma_s), numOfIter=int(num_iter)
        )
        return struct_u8.astype(np.float32) / 255.0, "u8"

structure, rgf_mode = compute_rgf(img, RGF_SIGMA_S, RGF_SIGMA_R, RGF_ITER)
stats("structure", structure)
save(structure, "01_structure.png")

# Get texture layer -> original image - structure layer
texture = img - structure
stats("texture (raw)", texture)
save(np.clip((texture + 0.5), 0.0, 1.0), "02_texture_vis_shifted.png")

# Enhance brightness of structure: HSV + two guided filters (f1,f2) -> weighted average at the end
structure_u8 = np.clip(structure * 255.0, 0, 255).astype(np.uint8)
hsv_u8 = cv2.cvtColor(structure_u8, cv2.COLOR_BGR2HSV)
hsv = hsv_u8.astype(np.float32) / 255.0
h, s, v = cv2.split(hsv)
stats("h,s,v (structure)", np.stack([h,s,v], axis=-1))
# Guided filters on v (normalized)
v_norm = v.astype(np.float32)
try:
    f1 = cv2.ximgproc.guidedFilter(v_norm, v_norm, R1, LAMBDA1).astype(np.float32)
    f2 = cv2.ximgproc.guidedFilter(f1, f1, R2, LAMBDA2).astype(np.float32)
except Exception:
    v_u8 = np.clip(v_norm * 255.0, 0, 255).astype(np.uint8)
    f1_u8 = cv2.ximgproc.guidedFilter(v_u8, v_u8, R1, LAMBDA1)
    f2_u8 = cv2.ximgproc.guidedFilter(f1_u8, f1_u8, R2, LAMBDA2)
    f1 = f1_u8.astype(np.float32) / 255.0
    f2 = f2_u8.astype(np.float32) / 255.0

illumination = 0.5 * (f1 + f2)  
stats("illumination", illumination)
save(illumination, "03_illumination.png")

# Weber-Fechner function fitting
s_mean = np.mean(s)
v_mean = np.mean(v)
v_enhanced = (v * (1.0 + s * v)) / (np.maximum(v, illumination) + (s_mean * v_mean) + EPS)
v_enhanced = np.clip(v_enhanced, 0.0, 1.0)
stats("v_enhanced", v_enhanced)

# Stretching the saturation
b_ch, g_ch, r_ch = cv2.split(structure)  
max_rgb = np.maximum(np.maximum(r_ch, g_ch), b_ch)
min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
mean_rgb = np.maximum(np.mean(structure, axis=2), 1e-6)
s_stretched = (0.5 + 0.5 * ((max_rgb + min_rgb + 1e-6) / (2 * mean_rgb + 1.0 + EPS))) * s
s_stretched = np.clip(s_stretched, 0.0, 1.0)
stats("s_stretched", s_stretched)

# merge hsv 
enhanced_hsv = cv2.merge([h, s_stretched, v_enhanced])
enhanced_struct_u8 = cv2.cvtColor((enhanced_hsv * 255.0).astype(np.uint8), cv2.COLOR_HSV2BGR)
enhanced_structure = enhanced_struct_u8.astype(np.float32) / 255.0
stats("enhanced_structure", enhanced_structure)
save(enhanced_structure, "04_enhanced_structure.png")

# Denoise texture layer: shift->gamma->BM3D(or NLMeans)->inv gamma->shift back
texture_shifted = np.clip(texture + 0.5, 0.0, 1.0)
stats("texture_shifted", texture_shifted)
save(texture_shifted, "05_texture_shifted.png")
# Gamma transform
texture_gamma_in = np.power(texture_shifted, 1.0 / GAMMA_TEXTURE)
save(texture_gamma_in, "06_texture_gamma_in.png")

# Denoising using BM3d
denoised_texture = None
print("Using bm3d for denoising")
denoised_texture = np.zeros_like(texture_gamma_in)
for ch in range(3):
    denoised_texture[..., ch] = bm3d(texture_gamma_in[..., ch].astype(np.float32), sigma_psd=0.05)
denoised_texture = np.power(np.clip(denoised_texture, 0.0, 1.0), GAMMA_TEXTURE) - 0.5
stats("denoised_texture", denoised_texture)
save(np.clip(denoised_texture + 0.5, 0.0, 1.0), "07_denoised_texture_vis.png")

# Enhanced structure + final texture
final_texture = denoised_texture
fused = enhanced_structure + final_texture
fused = np.clip(fused, 0.0, 1.0)
stats("fused (pre-edge)", fused)
save(fused, "08_fused.png")

# Edge enhancement on v channel
fused_u8 = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
h_f_u8, s_f_u8, v_f_u8 = cv2.split(cv2.cvtColor(fused_u8, cv2.COLOR_BGR2HSV))
h_f = h_f_u8.astype(np.float32) / 255.0
s_f = s_f_u8.astype(np.float32) / 255.0
v_f = v_f_u8.astype(np.float32) / 255.0

v_f_norm = v_f.copy()
# guidedFilter for q_v
try:
    q_v = cv2.ximgproc.guidedFilter(v_f_norm.astype(np.float32), v_f_norm.astype(np.float32), EDGE_RADIUS, EDGE_EPS)
except Exception:
    q_v = cv2.boxFilter(v_f_norm, ddepth=-1, ksize=(3,3))

ksize = (EDGE_RADIUS * 2 + 1, EDGE_RADIUS * 2 + 1)
mean_v = cv2.boxFilter(v_f_norm, ddepth=-1, ksize=ksize)
mean_vv = cv2.boxFilter(v_f_norm * v_f_norm, ddepth=-1, ksize=ksize)
var_v = np.maximum(mean_vv - mean_v * mean_v, 0.0)

a_k = var_v / (var_v + EDGE_EPS + EPS)
a_bar = cv2.boxFilter(a_k.astype(np.float32), ddepth=-1, ksize=ksize)
a_bar = np.clip(a_bar, 0.01, 0.99)
beta = np.power(a_bar / (1.0 - a_bar + EPS), GAMMA_EDGE)
beta = np.clip(beta, 0.0, BETA_CLAMP_MAX)

output_v = v_f_norm + beta * (v_f_norm - q_v)
output_v = np.clip(output_v, 0.0, 1.0)

final_hsv_u8 = cv2.merge([
    (h_f * 255.0).astype(np.uint8),
    (s_f * 255.0).astype(np.uint8),
    (output_v * 255.0).astype(np.uint8)
])
final_bgr_u8 = cv2.cvtColor(final_hsv_u8, cv2.COLOR_HSV2BGR)
final_bgr = final_bgr_u8.astype(np.float32) / 255.0
stats("final_bgr", final_bgr)
save(final_bgr, "09_final.png")

# plot
orig_rgb = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
final_rgb = cv2.cvtColor((final_bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(orig_rgb); plt.axis("off")
plt.subplot(1,2,2); plt.title("Enhanced (final)"); plt.imshow(final_rgb); plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_comparison.png"), dpi=150)
plt.show()

print("Saved all outputs to:", OUT_DIR)
