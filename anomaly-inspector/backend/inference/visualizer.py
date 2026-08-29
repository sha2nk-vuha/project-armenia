"""Rendering for inspection visualisations.

Heatmap/segmentation compositing for Anomaly Detection, plus the single
renderer that turns Decision Rules' declarative annotations into drawn
geometry. Every JPEG in the app is encoded by `encode_jpeg` so quality and
colour handling stay consistent.
"""
import cv2
import numpy as np

from inference.preprocessor import PreprocessTransform


def resize_anomaly_map_to_original(
    anomaly_map: np.ndarray,
    transform: PreprocessTransform | None,
    original_rgb: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Map an anomaly tensor from model space back onto the original image.

    When `transform` is provided the map is first resized to model resolution,
    the letterbox padding is cropped away, and the content is scaled to the
    original frame. Without a transform the map is stretched directly to the
    original size (legacy behaviour).
    """
    amap = anomaly_map.squeeze().astype(np.float32)
    orig_h, orig_w = original_rgb.shape[:2]

    if transform is None:
        if amap.shape != (orig_h, orig_w):
            return cv2.resize(amap, (orig_w, orig_h), interpolation=interpolation)
        return amap

    model_h, model_w = transform.model_hw
    if amap.shape != (model_h, model_w):
        amap = cv2.resize(amap, (model_w, model_h), interpolation=interpolation)

    scaled_h, scaled_w = transform.scaled_hw
    pad_top, pad_left = transform.pad_top, transform.pad_left
    cropped = amap[pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w]

    if cropped.shape != (orig_h, orig_w):
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=interpolation)
    return cropped


def generate_heatmap(
    anomaly_map: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.5,
    *,
    transform: PreprocessTransform | None = None,
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
        transform: Preprocess transform used when the image was fed to the model.

    Returns:
        JPEG bytes of the jet-coloured heatmap over the original.
    """
    amap = anomaly_map.squeeze().astype(np.float32)
    amap_resized = resize_anomaly_map_to_original(amap, transform, original_rgb)
    amap_norm = (np.clip(amap_resized, 0.0, 1.0) * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return encode_jpeg(blended_rgb)


def generate_segmentation(
    anomaly_map: np.ndarray,
    original_rgb: np.ndarray,
    threshold: float,
    *,
    transform: PreprocessTransform | None = None,
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
        transform: Preprocess transform used when the image was fed to the model.

    Returns:
        JPEG bytes of the original with red contours and a semi-transparent
        defect overlay.
    """
    amap = np.clip(anomaly_map.squeeze().astype(np.float32), 0.0, 1.0)
    amap_resized = resize_anomaly_map_to_original(amap, transform, original_rgb)

    mask = (amap_resized > threshold).astype(np.uint8) * 255

    result_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay = result_bgr.copy()
    overlay[mask == 255] = (0, 0, 200)
    result_bgr = cv2.addWeighted(result_bgr, 0.7, overlay, 0.3, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contours, -1, (0, 0, 255), 2)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return encode_jpeg(result_rgb)


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
