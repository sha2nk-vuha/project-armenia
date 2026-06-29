import cv2
import numpy as np


def generate_heatmap(
    anomaly_map: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.5
) -> bytes:
    """
    anomaly_map: float32 [1,1,H,W] or [H,W]
    original_rgb: uint8 [H,W,3]
    Returns PNG bytes of jet-coloured heatmap blended over the original.
    """
    amap = anomaly_map.squeeze().astype(np.float32)

    amap_min, amap_max = float(amap.min()), float(amap.max())
    if amap_max > amap_min:
        amap_norm = ((amap - amap_min) / (amap_max - amap_min) * 255).astype(np.uint8)
    else:
        amap_norm = np.zeros_like(amap, dtype=np.uint8)

    h, w = original_rgb.shape[:2]
    amap_resized = cv2.resize(amap_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_bgr = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

    _, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def generate_segmentation(
    anomaly_map: np.ndarray, original_rgb: np.ndarray, threshold: float
) -> bytes:
    """
    anomaly_map: float32 [1,1,H,W] or [H,W]
    original_rgb: uint8 [H,W,3]
    threshold: float in [0,1]; anomaly_map is normalised to [0,1] before comparison.
    Returns PNG bytes of original with red contour + semi-transparent defect overlay.
    """
    amap = anomaly_map.squeeze().astype(np.float32)

    amap_min, amap_max = float(amap.min()), float(amap.max())
    if amap_max > amap_min:
        amap_norm = (amap - amap_min) / (amap_max - amap_min)
    else:
        amap_norm = np.zeros_like(amap)

    h, w = original_rgb.shape[:2]
    amap_resized = cv2.resize(amap_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (amap_resized > threshold).astype(np.uint8) * 255

    result_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay = result_bgr.copy()
    overlay[mask == 255] = (0, 0, 200)
    result_bgr = cv2.addWeighted(result_bgr, 0.7, overlay, 0.3, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contours, -1, (0, 0, 255), 2)

    _, buf = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
