import React, { useState, useRef } from 'react';
import { Upload, Loader2, X, Info, Sparkles, Eye, Tag, Camera, Video, VideoOff } from 'lucide-react';

export default function NovelObjectDetection() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [detections, setDetections] = useState(null);
    const [error, setError] = useState(null);
    const [cameraActive, setCameraActive] = useState(false);
    const [stream, setStream] = useState(null);
    const fileInputRef = useRef(null);
    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const captureCanvasRef = useRef(null);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 10 * 1024 * 1024) {
                setError('File size must be less than 10MB');
                return;
            }

            setImage(file);
            setError(null);
            setDetections(null);

            const reader = new FileReader();
            reader.onload = (e) => setPreview(e.target.result);
            reader.readAsDataURL(file);
        }
    };

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            setStream(mediaStream);
            setCameraActive(true);
            setError(null);
            setPreview(null);
            setImage(null);
            setDetections(null);

            // Wait for video element to be ready
            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                }
            }, 100);
        } catch (err) {
            setError('Unable to access camera. Please ensure camera permissions are granted.');
            console.error('Camera error:', err);
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
        setCameraActive(false);
    };

    const captureImage = () => {
        if (!videoRef.current || !captureCanvasRef.current) return;

        const video = videoRef.current;
        const canvas = captureCanvasRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        // Convert canvas to blob
        canvas.toBlob((blob) => {
            if (blob) {
                const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
                setImage(file);
                setPreview(canvas.toDataURL('image/jpeg'));
                stopCamera();
            }
        }, 'image/jpeg', 0.95);
    };

    const simulateDetection = async () => {
        setLoading(true);
        setError(null);

        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 2500));

        // Simulate detection results
        const mockDetections = {
            known_objects: [
                { class: 'person', confidence: 0.95, bbox: [120, 80, 280, 420] },
                { class: 'chair', confidence: 0.89, bbox: [50, 300, 180, 480] },
                { class: 'laptop', confidence: 0.92, bbox: [200, 200, 380, 280] }
            ],
            novel_objects: [
                { class: 'cushion', confidence: 0.78, bbox: [300, 320, 420, 400] },
                { class: 'decorative vase', confidence: 0.73, bbox: [420, 150, 500, 280] },
                { class: 'wall art', confidence: 0.68, bbox: [80, 40, 200, 160] }
            ],
            metrics: {
                known_ap: 42.08,
                novel_ap: 17.42,
                total_detections: 6,
                processing_time: 2.3
            }
        };

        setDetections(mockDetections);
        setLoading(false);

        // Draw bounding boxes on canvas
        setTimeout(() => drawDetections(mockDetections), 100);
    };

    const drawDetections = (data) => {
        const canvas = canvasRef.current;
        const img = new Image();
        img.src = preview;

        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            ctx.drawImage(img, 0, 0);

            // Draw known objects (emerald/teal boxes)
            ctx.strokeStyle = '#14b8a6';
            ctx.lineWidth = 3;
            ctx.font = '16px Inter, sans-serif';

            data.known_objects.forEach(obj => {
                const [x1, y1, x2, y2] = obj.bbox;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = '#14b8a6';
                ctx.fillRect(x1, y1 - 25, 150, 25);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(`${obj.class} ${(obj.confidence * 100).toFixed(0)}%`, x1 + 5, y1 - 7);
            });

            // Draw novel objects (amber/gold boxes)
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;

            data.novel_objects.forEach(obj => {
                const [x1, y1, x2, y2] = obj.bbox;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = '#f59e0b';
                ctx.fillRect(x1, y1 - 25, 180, 25);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(`${obj.class} ${(obj.confidence * 100).toFixed(0)}%`, x1 + 5, y1 - 7);
            });
        };
    };

    const handleSubmit = async () => {
        if (!image) {
            setError('Please upload an image first');
            return;
        }

        await simulateDetection();
    };

    const resetApp = () => {
        stopCamera();
        setImage(null);
        setPreview(null);
        setDetections(null);
        setError(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 p-6">
            {/* Hidden canvas for capturing image */}
            <canvas ref={captureCanvasRef} className="hidden" />

            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center gap-2 bg-gradient-to-r from-cyan-500/20 to-teal-500/20 px-4 py-2 rounded-full mb-4 border border-cyan-500/30">
                        <Sparkles className="w-5 h-5 text-cyan-400" />
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-white via-cyan-200 to-teal-200 bg-clip-text text-transparent mb-3">
                        Novel Object Detection
                    </h1>
                    <p className="text-slate-300 text-lg max-w-2xl mx-auto">
                        Advanced system using CLIP, SAM, and Grounding DINO for detecting both known and novel objects
                    </p>
                </div>

                {/* Info Banner */}
                <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/30 rounded-xl p-4 mb-6 flex gap-3">
                    <Info className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-amber-100">
                        <strong className="text-amber-300">How it works:</strong> This system leverages cooperative foundational models (CLIP for classification, SAM for segmentation, GDINO for detection) to identify both familiar and previously unseen object categories without additional training.
                    </div>
                </div>

                <div className="grid lg:grid-cols-2 gap-6">
                    {/* Upload Section */}
                    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 shadow-xl shadow-black/20">
                        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                            <Upload className="w-5 h-5 text-cyan-400" />
                            Upload or Capture Image
                        </h2>

                        {/* Camera/Upload Toggle Buttons */}
                        {!preview && !cameraActive && (
                            <div className="grid grid-cols-2 gap-4 mb-4">
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    className="border-2 border-dashed border-cyan-500/40 rounded-xl p-8 text-center cursor-pointer hover:border-cyan-400 hover:bg-cyan-500/5 transition-all group"
                                >
                                    <Upload className="w-12 h-12 text-cyan-500 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                                    <p className="text-white font-medium mb-1">Upload Image</p>
                                    <p className="text-slate-400 text-xs">PNG, JPG up to 10MB</p>
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        onChange={handleImageUpload}
                                        className="hidden"
                                    />
                                </div>
                                <div
                                    onClick={startCamera}
                                    className="border-2 border-dashed border-amber-500/40 rounded-xl p-8 text-center cursor-pointer hover:border-amber-400 hover:bg-amber-500/5 transition-all group"
                                >
                                    <Camera className="w-12 h-12 text-amber-500 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                                    <p className="text-white font-medium mb-1">Capture Image</p>
                                    <p className="text-slate-400 text-xs">Use your camera</p>
                                </div>
                            </div>
                        )}

                        {/* Camera Preview */}
                        {cameraActive && (
                            <div className="relative">
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full rounded-xl shadow-lg border border-slate-700"
                                />
                                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-3">
                                    <button
                                        onClick={captureImage}
                                        className="bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-semibold py-3 px-6 rounded-full transition-all shadow-lg shadow-amber-500/25 flex items-center gap-2"
                                    >
                                        <Camera className="w-5 h-5" />
                                        Capture
                                    </button>
                                    <button
                                        onClick={stopCamera}
                                        className="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-3 px-6 rounded-full transition-all flex items-center gap-2"
                                    >
                                        <VideoOff className="w-5 h-5" />
                                        Cancel
                                    </button>
                                </div>
                                <div className="absolute top-4 left-4 bg-red-500 text-white text-xs px-3 py-1 rounded-full flex items-center gap-2 animate-pulse">
                                    <Video className="w-3 h-3" />
                                    LIVE
                                </div>
                            </div>
                        )}

                        {/* Image Preview */}
                        {preview && !cameraActive && (
                            <div className="relative">
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="w-full rounded-lg shadow-lg border border-slate-700"
                                />
                                <button
                                    onClick={resetApp}
                                    className="absolute top-2 right-2 bg-rose-500 hover:bg-rose-600 text-white p-2 rounded-full transition-colors shadow-lg"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 bg-rose-500/20 border border-rose-500/50 rounded-lg p-3 text-rose-200 text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            onClick={handleSubmit}
                            disabled={!image || loading}
                            className="w-full mt-6 bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600 disabled:from-slate-600 disabled:to-slate-700 text-white font-semibold py-3 px-6 rounded-xl transition-all disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Eye className="w-5 h-5" />
                                    Detect Objects
                                </>
                            )}
                        </button>
                    </div>

                    {/* Results Section */}
                    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 shadow-xl shadow-black/20">
                        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                            <Tag className="w-5 h-5 text-amber-400" />
                            Detection Results
                        </h2>

                        {!detections && !loading && (
                            <div className="text-center py-16 text-slate-400">
                                <Eye className="w-16 h-16 mx-auto mb-4 opacity-30" />
                                <p>Upload or capture an image and click detect to see results</p>
                            </div>
                        )}

                        {loading && (
                            <div className="text-center py-16">
                                <Loader2 className="w-16 h-16 mx-auto mb-4 text-cyan-400 animate-spin" />
                                <p className="text-slate-300">Analyzing image with AI models...</p>
                                <p className="text-slate-500 text-sm mt-2">CLIP + SAM + GDINO working together</p>
                            </div>
                        )}

                        {detections && (
                            <div className="space-y-4">
                                {/* Metrics */}
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-gradient-to-br from-teal-500/20 to-cyan-500/10 border border-teal-500/30 rounded-xl p-4">
                                        <p className="text-teal-300 text-sm font-medium">Known AP</p>
                                        <p className="text-3xl font-bold text-white">{detections.metrics.known_ap}</p>
                                    </div>
                                    <div className="bg-gradient-to-br from-amber-500/20 to-orange-500/10 border border-amber-500/30 rounded-xl p-4">
                                        <p className="text-amber-300 text-sm font-medium">Novel AP</p>
                                        <p className="text-3xl font-bold text-white">{detections.metrics.novel_ap}</p>
                                    </div>
                                </div>

                                {/* Detected Objects */}
                                <div>
                                    <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                                        <div className="w-3 h-3 bg-teal-500 rounded-full shadow-lg shadow-teal-500/50"></div>
                                        Known Objects ({detections.known_objects.length})
                                    </h3>
                                    <div className="space-y-2">
                                        {detections.known_objects.map((obj, idx) => (
                                            <div key={idx} className="bg-teal-500/10 border border-teal-500/30 rounded-lg p-3 hover:bg-teal-500/15 transition-colors">
                                                <div className="flex justify-between items-center">
                                                    <span className="text-white font-medium capitalize">{obj.class}</span>
                                                    <span className="text-teal-300 text-sm font-semibold">{(obj.confidence * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                                        <div className="w-3 h-3 bg-amber-500 rounded-full shadow-lg shadow-amber-500/50"></div>
                                        Novel Objects ({detections.novel_objects.length})
                                    </h3>
                                    <div className="space-y-2">
                                        {detections.novel_objects.map((obj, idx) => (
                                            <div key={idx} className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 hover:bg-amber-500/15 transition-colors">
                                                <div className="flex justify-between items-center">
                                                    <span className="text-white font-medium capitalize">{obj.class}</span>
                                                    <span className="text-amber-300 text-sm font-semibold">{(obj.confidence * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="text-slate-500 text-xs text-center pt-2">
                                    Processing time: {detections.metrics.processing_time}s
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Visualization */}
                {detections && (
                    <div className="mt-6 bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 shadow-xl shadow-black/20">
                        <h2 className="text-xl font-semibold text-white mb-4">Annotated Image</h2>
                        <div className="rounded-xl overflow-hidden border border-slate-700">
                            <canvas ref={canvasRef} className="w-full h-auto" />
                        </div>
                        <div className="flex gap-6 justify-center mt-4 text-sm">
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-teal-500 rounded"></div>
                                <span className="text-slate-300">Known Objects</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 border-2 border-amber-500 rounded"></div>
                                <span className="text-slate-300">Novel Objects</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Footer */}
                <div className="mt-8 text-center text-slate-500 text-sm">
                    <p>Based on "Enhancing Novel Object Detection via Cooperative Foundational Models" (WACV 2025)</p>
                </div>
            </div>
        </div>
    );
}