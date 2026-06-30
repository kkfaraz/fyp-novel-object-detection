import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Upload, Camera, VideoOff, Loader2, Eye, EyeOff, AlertCircle, RefreshCw,
    SlidersHorizontal, X, Sparkles, Clock, ChevronDown, Bug, Layers
} from 'lucide-react'
import { detectObjects, checkHealth } from '../utils/api'
import DetectionCanvas from '../components/DetectionCanvas'
import ControlPanel from '../components/ControlPanel'
import LoadingOverlay from '../components/LoadingOverlay'
import ErrorBanner from '../components/ErrorBanner'

export default function DetectionPage() {
    // Image state
    const [image, setImage] = useState(null)
    const [preview, setPreview] = useState(null)

    // Camera state
    const [cameraActive, setCameraActive] = useState(false)
    const [stream, setStream] = useState(null)
    const [fps, setFps] = useState(0)
    const [cameraError, setCameraError] = useState(null)
    const videoRef = useRef(null)
    const captureCanvasRef = useRef(null)
    const fpsIntervalRef = useRef(null)
    const frameCountRef = useRef(0)

    // Detection state
    const [loading, setLoading] = useState(false)
    const [currentStage, setCurrentStage] = useState(null)
    const [detections, setDetections] = useState(null)
    const [annotatedImage, setAnnotatedImage] = useState(null)
    const [stageImages, setStageImages] = useState(null)
    const [activeStageTab, setActiveStageTab] = useState('stage3')
    const [stageTimes, setStageTimes] = useState(null)
    const [error, setError] = useState(null)
    const [debugData, setDebugData] = useState(null)

    // Control state
    const [showKnown, setShowKnown] = useState(true)
    const [showUnknown, setShowUnknown] = useState(true)
    const [showMasks, setShowMasks] = useState(true)
    const [showScores, setShowScores] = useState(true)
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.1)
    const [debugMode, setDebugMode] = useState(false)

    // Backend health
    const [backendStatus, setBackendStatus] = useState(null)

    const fileInputRef = useRef(null)
    const cameraInputRef = useRef(null)

    // Check backend health on mount
    useEffect(() => {
        checkHealth().then(setBackendStatus)
    }, [])

    // Cleanup camera on unmount
    useEffect(() => {
        return () => {
            if (stream) stream.getTracks().forEach(t => t.stop())
            if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current)
        }
    }, [stream])

    // --- Image Upload ---
    const handleImageUpload = (e) => {
        const file = e.target.files?.[0]
        if (!file) return

        if (file.size > 10 * 1024 * 1024) {
            setError('File size must be less than 10MB')
            return
        }

        stopCamera()
        setImage(file)
        setError(null)
        setDetections(null)
        setAnnotatedImage(null)
        setDebugData(null)

        const reader = new FileReader()
        reader.onload = (e) => setPreview(e.target.result)
        reader.readAsDataURL(file)
    }

    // --- Camera Logic ---
    const startCamera = async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setCameraError('Camera not supported or permission denied. Please upload an image instead.')
            if (cameraInputRef.current) cameraInputRef.current.click()
            return
        }

        try {
            let mediaStream = null
            const constraints = [
                { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } },
                { video: { facingMode: 'user' } },
                { video: true },
            ]

            for (const constraint of constraints) {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia(constraint)
                    break
                } catch { continue }
            }

            if (!mediaStream) throw new Error('Could not access any camera')

            setStream(mediaStream)
            setCameraActive(true)
            setCameraError(null)
            setError(null)
            setPreview(null)
            setImage(null)
            setDetections(null)
            setAnnotatedImage(null)
            setDebugData(null)

            // FPS counter
            frameCountRef.current = 0
            fpsIntervalRef.current = setInterval(() => {
                setFps(frameCountRef.current)
                frameCountRef.current = 0
            }, 2000)

            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream
                    videoRef.current.play().catch(() => { })

                    // Track frames
                    const trackFps = () => {
                        frameCountRef.current++
                        if (videoRef.current && !videoRef.current.paused) {
                            requestAnimationFrame(trackFps)
                        }
                    }
                    requestAnimationFrame(trackFps)
                }
            }, 200)
        } catch (err) {
            console.error('Camera error:', err)
            setCameraError('Camera access denied or not available. Please upload an image instead.')
            if (cameraInputRef.current) cameraInputRef.current.click()
        }
    }

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(t => t.stop())
            setStream(null)
        }
        setCameraActive(false)
        setCameraError(null)
        if (fpsIntervalRef.current) {
            clearInterval(fpsIntervalRef.current)
            fpsIntervalRef.current = null
        }
        setFps(0)
    }, [stream])

    const captureImage = () => {
        if (!videoRef.current || !captureCanvasRef.current) return

        const video = videoRef.current
        const canvas = captureCanvasRef.current
        const ctx = canvas.getContext('2d')

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)

        canvas.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' })
                setImage(file)
                setPreview(canvas.toDataURL('image/jpeg'))
                stopCamera()
            }
        }, 'image/jpeg', 0.95)
    }

    // --- Detection ---
    const runDetection = async () => {
        if (!image) {
            setError('Please upload or capture an image first')
            return
        }

        setLoading(true)
        setError(null)
        setCurrentStage('stage1')

        // Simulate stage progression for UX (actual timing comes from backend)
        const stageTimer = setTimeout(() => setCurrentStage('stage2'), 3000)
        const stageTimer2 = setTimeout(() => setCurrentStage('stage3'), 8000)

        try {
            const data = await detectObjects(image)

            clearTimeout(stageTimer)
            clearTimeout(stageTimer2)
            setCurrentStage(null)

            setDetections(data.detections || [])
            setAnnotatedImage(data.annotated_image || null)
            setStageImages(data.stage_images || null)
            // Default to combined if available, else stage3
            if (data.stage_images && data.stage_images.combined) {
                setActiveStageTab('combined')
            } else {
                setActiveStageTab('stage3')
            }
            setStageTimes(data.stage_times || null)
            setDebugData(data.debug || null)

        } catch (err) {
            clearTimeout(stageTimer)
            clearTimeout(stageTimer2)
            setCurrentStage(null)
            console.error('Detection error:', err)
            setError(err.message || 'Detection failed. Make sure the backend server is running.')
        } finally {
            setLoading(false)
        }
    }

    const resetAll = () => {
        stopCamera()
        setImage(null)
        setPreview(null)
        setDetections(null)
        setAnnotatedImage(null)
        setStageImages(null)
        setActiveStageTab('combined')
        setStageTimes(null)
        setError(null)
        setDebugData(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
        if (cameraInputRef.current) cameraInputRef.current.value = ''
    }

    // Filtered detections for display
    const filteredDetections = (detections || []).filter(d => {
        if (d.confidence < confidenceThreshold) return false
        if (d.type === 'known' && !showKnown) return false
        if (d.type === 'unknown' && !showUnknown) return false
        return true
    })

    const knownCount = filteredDetections.filter(d => d.type === 'known').length
    const unknownCount = filteredDetections.filter(d => d.type === 'unknown').length

    return (
        <div className="relative overflow-hidden min-h-screen">
            <canvas ref={captureCanvasRef} className="hidden" />
            <input ref={cameraInputRef} type="file" accept="image/*" capture="environment" onChange={handleImageUpload} className="hidden" />

            {/* Background */}
            <div className="absolute top-20 left-1/4 w-80 h-80 bg-primary-500/5 rounded-full blur-3xl pointer-events-none" />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-8 pb-24">
                {/* Header */}
                <motion.div
                    className="text-center mb-8"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1 className="text-3xl sm:text-4xl font-extrabold text-white mb-2">
                        Detection{' '}
                        <span className="bg-gradient-to-r from-primary-400 to-emerald-400 bg-clip-text text-transparent">
                            Interface
                        </span>
                    </h1>
                    <p className="text-surface-400 text-sm max-w-xl mx-auto">
                        Upload an image or capture from camera. The 3-stage pipeline will detect known and novel objects.
                    </p>

                    {/* Backend Status */}
                    {backendStatus && (
                        <div className={`inline-flex items-center gap-2 mt-3 px-3 py-1 rounded-full text-xs font-medium ${(backendStatus.status === 'online' || backendStatus.status === 'idle')
                            ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                            : backendStatus.status === 'loading'
                            ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                            : backendStatus.status === 'error'
                            ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                            : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                            }`}>
                            <div className={`w-2 h-2 rounded-full ${(backendStatus.status === 'online' || backendStatus.status === 'idle')
                                ? 'bg-emerald-400 animate-pulse'
                                : backendStatus.status === 'loading'
                                ? 'bg-blue-400 animate-pulse'
                                : backendStatus.status === 'error'
                                ? 'bg-rose-400'
                                : 'bg-amber-400'}`} />
                            {(backendStatus.status === 'online' || backendStatus.status === 'idle') && 'Backend Online'}
                            {backendStatus.status === 'loading' && 'Backend Loading...'}
                            {backendStatus.status === 'error' && 'Backend Error'}
                            {backendStatus.status === 'offline' && 'Backend Offline'}
                        </div>
                    )}
                </motion.div>

                <div className="grid lg:grid-cols-5 gap-6">
                    {/* Left Panel: Input + Controls */}
                    <div className="lg:col-span-2 space-y-4">
                        {/* Input Section */}
                        <motion.div
                            className="glass-card p-5"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.1 }}
                        >
                            <h2 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
                                <Upload className="w-4 h-4 text-primary-400" />
                                Input Image
                            </h2>

                            {/* Camera/Upload Options */}
                            {!preview && !cameraActive && (
                                <div className="grid grid-cols-2 gap-3">
                                    <button
                                        onClick={() => fileInputRef.current?.click()}
                                        className="border border-dashed border-surface-600 rounded-xl p-6 text-center hover:border-primary-500/50 hover:bg-primary-500/5 transition-all group"
                                    >
                                        <Upload className="w-8 h-8 text-surface-500 mx-auto mb-2 group-hover:text-primary-400 group-hover:scale-110 transition-all" />
                                        <p className="text-white text-sm font-medium">Upload</p>
                                        <p className="text-surface-500 text-xs mt-0.5">PNG, JPG ≤10MB</p>
                                        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                                    </button>

                                    <button
                                        onClick={startCamera}
                                        className="border border-dashed border-surface-600 rounded-xl p-6 text-center hover:border-amber-500/50 hover:bg-amber-500/5 transition-all group"
                                    >
                                        <Camera className="w-8 h-8 text-surface-500 mx-auto mb-2 group-hover:text-amber-400 group-hover:scale-110 transition-all" />
                                        <p className="text-white text-sm font-medium">Camera</p>
                                        <p className="text-surface-500 text-xs mt-0.5">Live capture</p>
                                    </button>
                                </div>
                            )}

                            {/* Camera Error */}
                            {cameraError && (
                                <div className="mt-3 bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 text-amber-300 text-xs flex items-start gap-2">
                                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                                    {cameraError}
                                </div>
                            )}

                            {/* Camera Preview */}
                            {cameraActive && (
                                <div className="relative rounded-xl overflow-hidden">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full rounded-xl border border-surface-700"
                                    />
                                    {/* FPS Counter */}
                                    <div className="absolute top-3 right-3 bg-black/70 text-white text-xs px-2 py-1 rounded-md font-mono">
                                        {fps} FPS
                                    </div>
                                    {/* LIVE badge */}
                                    <div className="absolute top-3 left-3 bg-red-500 text-white text-xs px-2.5 py-1 rounded-full flex items-center gap-1.5 animate-pulse font-medium">
                                        <div className="w-1.5 h-1.5 bg-white rounded-full" />
                                        LIVE
                                    </div>
                                    {/* Controls */}
                                    <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2">
                                        <button
                                            onClick={captureImage}
                                            className="bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-white font-medium py-2 px-5 rounded-full text-sm flex items-center gap-1.5 shadow-lg"
                                        >
                                            <Camera className="w-4 h-4" /> Capture
                                        </button>
                                        <button
                                            onClick={stopCamera}
                                            className="bg-surface-700/90 hover:bg-surface-600 text-white py-2 px-4 rounded-full text-sm flex items-center gap-1.5"
                                        >
                                            <VideoOff className="w-4 h-4" /> Stop
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Image Preview */}
                            {preview && !cameraActive && (
                                <div className="relative">
                                    <img
                                        src={preview}
                                        alt="Input preview"
                                        className="w-full rounded-xl border border-surface-700"
                                    />
                                    <button
                                        onClick={resetAll}
                                        className="absolute top-2 right-2 bg-rose-500 hover:bg-rose-600 text-white p-1.5 rounded-full transition-colors shadow-lg"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                            )}

                            {/* Error */}
                            {error && <ErrorBanner message={error} onRetry={image ? runDetection : null} onDismiss={() => setError(null)} />}

                            {/* Detect Button */}
                            <button
                                onClick={runDetection}
                                disabled={!image || loading}
                                className="w-full mt-4 bg-gradient-to-r from-primary-500 to-emerald-500 hover:from-primary-400 hover:to-emerald-400 disabled:from-surface-700 disabled:to-surface-700 text-white font-semibold py-3 rounded-xl transition-all disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-primary-500/20 hover:shadow-primary-500/35 disabled:shadow-none text-sm"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <Sparkles className="w-4 h-4" />
                                        Detect Objects
                                    </>
                                )}
                            </button>
                        </motion.div>

                        {/* Controls */}
                        {detections && (
                            <ControlPanel
                                showKnown={showKnown}
                                setShowKnown={setShowKnown}
                                showUnknown={showUnknown}
                                setShowUnknown={setShowUnknown}
                                showMasks={showMasks}
                                setShowMasks={setShowMasks}
                                showScores={showScores}
                                setShowScores={setShowScores}
                                confidenceThreshold={confidenceThreshold}
                                setConfidenceThreshold={setConfidenceThreshold}
                                debugMode={debugMode}
                                setDebugMode={setDebugMode}
                            />
                        )}

                        {/* Stage Times */}
                        {stageTimes && (
                            <motion.div
                                className="glass-card p-4"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <h3 className="text-white font-semibold text-xs mb-3 flex items-center gap-2">
                                    <Clock className="w-3.5 h-3.5 text-primary-400" />
                                    Inference Time
                                </h3>
                                <div className="space-y-2">
                                    {Object.entries(stageTimes).map(([stage, time]) => (
                                        <div key={stage} className="flex items-center justify-between">
                                            <span className="text-surface-400 text-xs capitalize">{stage.replace('_', ' ')}</span>
                                            <span className="text-white text-xs font-mono">{typeof time === 'number' ? `${time.toFixed(2)}s` : time}</span>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </div>

                    {/* Right Panel: Results */}
                    <div className="lg:col-span-3 space-y-4">
                        {/* Loading Overlay */}
                        {loading && <LoadingOverlay currentStage={currentStage} />}

                        {/* No results yet */}
                        {!loading && !detections && !annotatedImage && (
                            <motion.div
                                className="glass-card p-12 text-center"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.2 }}
                            >
                                <Eye className="w-16 h-16 text-surface-700 mx-auto mb-4" />
                                <p className="text-surface-500 text-sm">Upload an image and click detect to see results</p>
                            </motion.div>
                        )}

                        {/* Detection Results */}
                        {!loading && detections && (
                            <motion.div
                                className="space-y-4"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                {/* Stats Bar */}
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="glass-card p-3 text-center">
                                        <p className="text-2xl font-bold text-white">{filteredDetections.length}</p>
                                        <p className="text-surface-500 text-xs">Total</p>
                                    </div>
                                    <div className="glass-card p-3 text-center border-emerald-500/20">
                                        <p className="text-2xl font-bold text-emerald-400">{knownCount}</p>
                                        <p className="text-surface-500 text-xs">Known</p>
                                    </div>
                                    <div className="glass-card p-3 text-center border-amber-500/20">
                                        <p className="text-2xl font-bold text-amber-400">{unknownCount}</p>
                                        <p className="text-surface-500 text-xs">Unknown</p>
                                    </div>
                                </div>

                                {/* Original Image */}
                                {preview && (
                                    <div className="glass-card p-4">
                                        <h3 className="text-white font-semibold text-xs mb-3">Original Image</h3>
                                        <img src={preview} alt="Original" className="w-full rounded-lg border border-surface-700" />
                                    </div>
                                )}

                                {/* Detection Canvas / Annotated Image */}
                                <div className="glass-card p-4">
                                    <h3 className="text-white font-semibold text-xs mb-3 flex items-center gap-2">
                                        <Layers className="w-3.5 h-3.5 text-primary-400" />
                                        Detection Pipeline Results
                                    </h3>

                                    {/* Stage Tabs */}
                                    {stageImages && (
                                        <div className="flex gap-2 mb-4 bg-surface-800/50 p-1 rounded-xl w-fit">
                                            {[
                                                { id: 'stage1', label: 'Stage 1 (GDINO+RPN)' },
                                                { id: 'stage2', label: 'Stage 2 (VLRM)' },
                                                { id: 'stage3', label: 'Stage 3 (Final)' },
                                                { id: 'combined', label: 'Combined (All Stages)' }
                                            ].map((tab) => (
                                                <button
                                                    key={tab.id}
                                                    onClick={() => setActiveStageTab(tab.id)}
                                                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${activeStageTab === tab.id
                                                        ? 'bg-primary-500 text-white shadow-md'
                                                        : 'text-surface-400 hover:text-white hover:bg-surface-700'
                                                        }`}
                                                >
                                                    {tab.label}
                                                </button>
                                            ))}
                                        </div>
                                    )}

                                    {stageImages?.[activeStageTab] ? (
                                        <div className="relative">
                                            <img
                                                src={stageImages[activeStageTab].startsWith('data:') ? stageImages[activeStageTab] : `data:image/jpeg;base64,${stageImages[activeStageTab]}`}
                                                alt={`${activeStageTab} results`}
                                                className="w-full rounded-lg border border-surface-700"
                                            />
                                            <div className="absolute top-2 left-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded backdrop-blur-sm">
                                                {activeStageTab === 'stage1' && 'Knowns (Green) + Unknown ROIs'}
                                                {activeStageTab === 'stage2' && 'Novel Object Candidates (Blue)'}
                                                {activeStageTab === 'stage3' && 'Refined & Filtered Detections (Magenta)'}
                                                {activeStageTab === 'combined' && 'Full Pipeline Dashboard (All Results Overlayed)'}
                                            </div>
                                        </div>
                                    ) : annotatedImage && activeStageTab === 'stage3' ? (
                                        <img
                                            src={annotatedImage.startsWith('data:') ? annotatedImage : `data:image/jpeg;base64,${annotatedImage}`}
                                            alt="Detection results"
                                            className="w-full rounded-lg border border-surface-700"
                                        />
                                    ) : preview ? (
                                        <DetectionCanvas
                                            imageSrc={preview}
                                            detections={filteredDetections}
                                            showMasks={showMasks}
                                            showScores={showScores}
                                        />
                                    ) : null}

                                    {/* Legend */}
                                    <div className="flex flex-wrap gap-4 justify-center mt-3 text-xs">
                                        <div className="flex items-center gap-1.5">
                                            <div className="w-3 h-3 border-2 border-emerald-500 rounded-sm" />
                                            <span className="text-surface-400">Known</span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <div className="w-3 h-3 border-2 border-amber-500 rounded-sm" />
                                            <span className="text-surface-400">Unknown</span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <div className="w-3 h-3 border-2 border-red-400 rounded-sm" />
                                            <span className="text-surface-400">Low Confidence</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Detection List */}
                                <div className="glass-card p-4">
                                    <h3 className="text-white font-semibold text-xs mb-3">
                                        Detections ({filteredDetections.length})
                                    </h3>
                                    {filteredDetections.length === 0 ? (
                                        <p className="text-surface-500 text-xs text-center py-4">No detections matching current filters</p>
                                    ) : (
                                        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                                            {filteredDetections.map((det, idx) => (
                                                <div
                                                    key={det.id ?? idx}
                                                    className={`flex items-center justify-between p-2.5 rounded-lg text-sm transition-colors ${det.type === 'known'
                                                        ? 'bg-emerald-500/8 border border-emerald-500/20 hover:bg-emerald-500/12'
                                                        : 'bg-amber-500/8 border border-amber-500/20 hover:bg-amber-500/12'
                                                        }`}
                                                >
                                                    <div className="flex items-center gap-2.5 min-w-0">
                                                        <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${det.type === 'known'
                                                            ? 'bg-emerald-500/20 text-emerald-400'
                                                            : 'bg-amber-500/20 text-amber-400'
                                                            }`}>
                                                            {det.type === 'known' ? 'K' : 'U'}
                                                        </span>
                                                        <span className="text-white font-medium text-xs truncate capitalize">
                                                            {det.label || 'object'}
                                                        </span>
                                                    </div>
                                                    <span className={`text-xs font-mono font-semibold ${det.confidence >= 0.7 ? 'text-emerald-400' :
                                                        det.confidence >= 0.4 ? 'text-amber-400' : 'text-red-400'
                                                        }`}>
                                                        {(det.confidence * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Debug Mode */}
                                {debugMode && debugData && (
                                    <motion.div
                                        className="glass-card p-4"
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                    >
                                        <h3 className="text-white font-semibold text-xs mb-3 flex items-center gap-2">
                                            <Bug className="w-3.5 h-3.5 text-amber-400" />
                                            Debug Info
                                        </h3>
                                        <pre className="text-surface-400 text-xs font-mono bg-surface-900 rounded-lg p-3 overflow-x-auto max-h-48 overflow-y-auto">
                                            {JSON.stringify(debugData, null, 2)}
                                        </pre>
                                    </motion.div>
                                )}
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
