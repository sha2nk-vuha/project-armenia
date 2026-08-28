"""Rendering for inspection visualisations.

Heatmap/segmentation compositing for Anomaly Detection, plus the single
renderer that turns Decision Rules' declarative annotations into drawn
geometry. Every JPEG in the app is encoded by `encode_jpeg` so quality and
colour handling stay consistent.
"""
import cv2
import numpy as np


def generate_heatmap(
    anomaly_map: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.5
) -> bytes:
    """
    anomaly_map: float32 [1,1,H,W] or [H,W], already normalised to [0,1]
        (Anomalib bakes min-max + threshold normalisation into the exported
        graph, with 0.5 == the calibrated threshold).
    original_rgb: uint8 [H,W,3]
    Returns JPEG bytes of jet-coloured heatmap blended over the original.

    Uses a fixed [0,1]→[0,255] scale, NOT per-image min-max. Per-image
    normalisation would stretch a clean image's narrow score range across the
    full colour map and make every pixel look anomalous.

    Args:
        anomaly_map: Float map already normalised to [0,1]; [1,1,H,W] or [H,W].
        original_rgb: Frame to blend onto ([H,W,3] uint8).
        alpha: Heatmap opacity in the blend.

    Returns:
        JPEG bytes of the jet-coloured heatmap over the original.
    """
    amap = anomaly_map.squeeze().astype(np.float32)
    amap_norm = (np.clip(amap, 0.0, 1.0) * 255).astype(np.uint8)

    h, w = original_rgb.shape[:2]
    amap_resized = cv2.resize(amap_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_bgr = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

    return encode_jpeg(blended)


def generate_segmentation(
    anomaly_map: np.ndarray, original_rgb: np.ndarray, threshold: float
) -> bytes:
    """
    anomaly_map: float32 [1,1,H,W] or [H,W], already normalised to [0,1]
        (Anomalib bakes normalisation into the exported graph).
    original_rgb: uint8 [H,W,3]
    Args:
        anomaly_map: Float map already normalised to [0,1]; [1,1,H,W] or [H,W].
        original_rgb: Frame to overlay ([H,W,3] uint8).
        threshold: Cut-off in [0,1] — pixels scoring above it are defects.
            Operates on the same normalised scale as the image-level pred_score,
            so the segmentation agrees with the OK/NOK Verdict.

    Returns:
        JPEG bytes of the original with red contours and a semi-transparent
        defect overlay.
    """
    amap = np.clip(anomaly_map.squeeze().astype(np.float32), 0.0, 1.0)

    h, w = original_rgb.shape[:2]
    amap_resized = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (amap_resized > threshold).astype(np.uint8) * 255

    result_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay = result_bgr.copy()
    overlay[mask == 255] = (0, 0, 200)
    result_bgr = cv2.addWeighted(result_bgr, 0.7, overlay, 0.3, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contours, -1, (0, 0, 255), 2)

    return encode_jpeg(result_bgr)


def render_annotations(base_rgb: np.ndarray, annotations: list) -> np.ndarray:
    """Draw a Decision Rule's declarative annotations onto an RGB image.

    Rules return geometry (`Circle`, `Line`, `Point`, `Polygon`, `Text`) rather
    than rendered bytes, so drawing style lives here and stays consistent across
    rules.

    Args:
        base_rgb: Canvas to draw onto ([H,W,3] uint8).
        annotations: Declarative shapes produced by a Decision Rule.

    Returns:
        A new RGB array; empty annotation lists return the frame unchanged.
    """
    from inference.decision.base import Circle, Line, Point, Polygon, Text

    if not annotations:
        return base_rgb
    canvas = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR).copy()

    def bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
        """OpenCV wants BGR; rules speak RGB.

        Args:
            color: RGB tuple as rules declare it.

        Returns:
            The same colour as a BGR tuple for OpenCV drawing calls.
        """
        r, g, b = color
        return (b, g, r)

    for a in annotations:
        if isinstance(a, Point):
            cv2.circle(
                canvas, (int(round(a.x)), int(round(a.y))), a.radius, bgr(a.color),
                -1 if a.filled else 1, cv2.LINE_AA,
            )
        elif isinstance(a, Circle):
            cv2.circle(
                canvas, (int(round(a.cx)), int(round(a.cy))), int(round(a.r)),
                bgr(a.color), a.thickness, cv2.LINE_AA,
            )
        elif isinstance(a, Line):
            cv2.line(
                canvas, (int(round(a.x1)), int(round(a.y1))),
                (int(round(a.x2)), int(round(a.y2))), bgr(a.color), a.thickness,
                cv2.LINE_AA,
            )
        elif isinstance(a, Polygon):
            pts = np.array([[int(round(x)), int(round(y))] for x, y in a.points], dtype=np.int32)
            cv2.polylines(canvas, [pts], a.closed, bgr(a.color), a.thickness, cv2.LINE_AA)
        elif isinstance(a, Text):
            cv2.putText(
                canvas, a.text, (int(round(a.x)), int(round(a.y))),
                cv2.FONT_HERSHEY_SIMPLEX, a.scale, bgr(a.color), 1, cv2.LINE_AA,
            )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def encode_jpeg(image_rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB image as JPEG bytes.

    The single encode path for every visualisation the app produces, so JPEG
    quality policy lives here alone.

    Args:
        image_rgb: Image to encode ([H,W,3] uint8).
        quality: JPEG quality passed to OpenCV.

    Returns:
        JPEG-encoded bytes.
    """
    bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
