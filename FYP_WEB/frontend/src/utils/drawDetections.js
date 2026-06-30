/**
 * Draw detection bounding boxes and labels on a canvas.
 * 
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} detections - Array of detection objects
 * @param {Object} options - Rendering options
 */
export function drawDetections(ctx, detections, options = {}) {
    const {
        showKnown = true,
        showUnknown = true,
        showScores = true,
        showMasks = true,
        confidenceThreshold = 0.0,
        imageWidth = ctx.canvas.width,
        imageHeight = ctx.canvas.height,
        scaleX = 1,
        scaleY = 1,
    } = options;

    const filtered = detections.filter(d => {
        if (d.confidence < confidenceThreshold) return false;
        if (d.type === 'known' && !showKnown) return false;
        if (d.type === 'unknown' && !showUnknown) return false;
        return true;
    });

    // Draw masks first (underneath boxes)
    if (showMasks) {
        filtered.forEach(det => {
            if (det.mask) {
                drawMask(ctx, det.mask, det.bbox, scaleX, scaleY, det.type);
            }
        });
    }

    // Draw bounding boxes and labels
    filtered.forEach(det => {
        drawBox(ctx, det, scaleX, scaleY, showScores);
    });
}

function drawBox(ctx, det, scaleX, scaleY, showScores) {
    const [x1, y1, x2, y2] = det.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const w = sx2 - sx1;
    const h = sy2 - sy1;

    // Colors based on type and confidence
    let color, bgColor;
    if (det.type === 'known') {
        color = '#10b981'; // green
        bgColor = 'rgba(16, 185, 129, 0.15)';
    } else {
        color = '#f97316'; // orange
        bgColor = 'rgba(249, 115, 22, 0.15)';
    }

    if (det.confidence < 0.3) {
        color = '#f87171'; // light red for low confidence
    }

    // Draw box fill
    ctx.fillStyle = bgColor;
    ctx.fillRect(sx1, sy1, w, h);

    // Draw box border
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.strokeRect(sx1, sy1, w, h);

    // Draw label
    const label = det.label || 'object';
    const badge = det.type === 'known' ? 'K' : 'U';
    const scoreText = showScores ? ` ${(det.confidence * 100).toFixed(0)}%` : '';
    const text = `[${badge}] ${label}${scoreText}`;

    ctx.font = 'bold 12px Inter, sans-serif';
    const textMetrics = ctx.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = 18;
    const padding = 4;

    // Label background
    const labelY = sy1 - textHeight - padding > 0 ? sy1 - textHeight - padding : sy1;
    ctx.fillStyle = color;
    ctx.fillRect(sx1, labelY, textWidth + padding * 2, textHeight + padding);

    // Label text
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText(text, sx1 + padding, labelY + textHeight - 2);
}

function drawMask(ctx, maskData, bbox, scaleX, scaleY, type) {
    // maskData is a base64 PNG or an RLE mask
    // For simplicity, draw a semi-transparent fill over the bbox area
    const [x1, y1, x2, y2] = bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    const maskColor = type === 'known'
        ? 'rgba(16, 185, 129, 0.25)'
        : 'rgba(249, 115, 22, 0.25)';

    ctx.fillStyle = maskColor;
    ctx.fillRect(sx1, sy1, w, h);
}

/**
 * Get the tooltip info for a detection at a given canvas position.
 */
export function getDetectionAtPoint(detections, x, y, scaleX = 1, scaleY = 1) {
    // Search in reverse order so topmost drawn box wins
    for (let i = detections.length - 1; i >= 0; i--) {
        const det = detections[i];
        const [bx1, by1, bx2, by2] = det.bbox;
        const sx1 = bx1 * scaleX;
        const sy1 = by1 * scaleY;
        const sx2 = bx2 * scaleX;
        const sy2 = by2 * scaleY;

        if (x >= sx1 && x <= sx2 && y >= sy1 && y <= sy2) {
            return det;
        }
    }
    return null;
}
