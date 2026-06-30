import { useRef, useEffect, useState, useCallback } from 'react'
import { drawDetections, getDetectionAtPoint } from '../utils/drawDetections'

export default function DetectionCanvas({ imageSrc, detections, showMasks, showScores }) {
    const canvasRef = useRef(null)
    const imgRef = useRef(null)
    const containerRef = useRef(null)
    const [tooltip, setTooltip] = useState(null)
    const [imgLoaded, setImgLoaded] = useState(false)
    const [dimensions, setDimensions] = useState({ width: 0, height: 0, scaleX: 1, scaleY: 1 })

    // Load image
    useEffect(() => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
            imgRef.current = img
            setImgLoaded(true)
        }
        img.src = imageSrc
    }, [imageSrc])

    // Draw on canvas
    useEffect(() => {
        if (!imgLoaded || !canvasRef.current || !imgRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const img = imgRef.current

        // Calculate display size (fit container width)
        const container = containerRef.current
        const containerWidth = container ? container.clientWidth : 800
        const scale = containerWidth / img.naturalWidth
        const displayWidth = containerWidth
        const displayHeight = img.naturalHeight * scale

        canvas.width = displayWidth
        canvas.height = displayHeight

        const scaleX = displayWidth / img.naturalWidth
        const scaleY = displayHeight / img.naturalHeight

        setDimensions({ width: displayWidth, height: displayHeight, scaleX, scaleY })

        // Draw image
        ctx.clearRect(0, 0, displayWidth, displayHeight)
        ctx.drawImage(img, 0, 0, displayWidth, displayHeight)

        // Draw detections
        if (detections && detections.length > 0) {
            drawDetections(ctx, detections, {
                showKnown: true,
                showUnknown: true,
                showScores,
                showMasks,
                confidenceThreshold: 0,
                scaleX,
                scaleY,
            })
        }
    }, [imgLoaded, detections, showMasks, showScores])

    // Tooltip on hover
    const handleMouseMove = useCallback((e) => {
        if (!canvasRef.current || !detections) return

        const rect = canvasRef.current.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top

        const det = getDetectionAtPoint(detections, x, y, dimensions.scaleX, dimensions.scaleY)

        if (det) {
            setTooltip({
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
                detection: det,
            })
        } else {
            setTooltip(null)
        }
    }, [detections, dimensions])

    const handleMouseLeave = () => setTooltip(null)

    return (
        <div ref={containerRef} className="relative">
            <canvas
                ref={canvasRef}
                className="w-full rounded-lg border border-surface-700 cursor-crosshair"
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
            />

            {/* Tooltip */}
            {tooltip && (
                <div
                    className="absolute z-50 pointer-events-none animate-slide-down"
                    style={{
                        left: Math.min(tooltip.x + 12, (containerRef.current?.clientWidth || 400) - 200),
                        top: tooltip.y - 10,
                    }}
                >
                    <div className="bg-surface-900/95 backdrop-blur-lg border border-surface-600 rounded-lg p-3 shadow-xl min-w-[160px]">
                        <p className="text-white font-semibold text-xs capitalize mb-1">{tooltip.detection.label || 'object'}</p>
                        <div className="space-y-0.5 text-[10px]">
                            <div className="flex justify-between">
                                <span className="text-surface-500">Type</span>
                                <span className={tooltip.detection.type === 'known' ? 'text-emerald-400' : 'text-amber-400'}>
                                    {tooltip.detection.type}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-surface-500">Confidence</span>
                                <span className="text-white">{(tooltip.detection.confidence * 100).toFixed(1)}%</span>
                            </div>
                            {tooltip.detection.detector_source && (
                                <div className="flex justify-between">
                                    <span className="text-surface-500">Source</span>
                                    <span className="text-surface-300">{tooltip.detection.detector_source}</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
