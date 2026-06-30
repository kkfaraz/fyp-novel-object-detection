import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Search, Tag, Layers, ArrowDown, ChevronDown, ChevronUp, ArrowRight,
    Box, Scan, Brain, Shapes, Shuffle, Shield, Sparkles
} from 'lucide-react'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i = 0) => ({
        opacity: 1, y: 0,
        transition: { delay: i * 0.15, duration: 0.5, ease: 'easeOut' },
    }),
}

const stages = [
    {
        id: 1,
        title: 'Stage 1: Initialization (Detection)',
        icon: Search,
        color: 'from-cyan-500 to-blue-600',
        borderColor: 'border-cyan-500/40',
        glowColor: 'shadow-cyan-500/20',
        accentBg: 'bg-cyan-500/10',
        accentText: 'text-cyan-400',
        models: [
            { name: 'GroundingDINO', detail: 'Open-vocabulary detection using text prompts for known + candidate novel categories' },
            { name: 'Mask R-CNN (R101-FPN)', detail: 'Class-agnostic proposals preserving background ROIs for unknown discovery' },
        ],
        inputs: ['Input image', 'Text prompts (known class names + candidate novel)'],
        outputs: ['Known object detections with bounding boxes', 'Background ROIs (potential unknown objects)'],
        details: [
            'GroundingDINO uses Swin-T backbone with text-conditioned detection heads',
            'Mask R-CNN runs in parallel to generate class-agnostic region proposals',
            'Background ROIs are kept — they may contain novel objects',
            'Both detectors run simultaneously for efficiency',
        ],
    },
    {
        id: 2,
        title: 'Stage 2: Unknown Object Labelling',
        icon: Tag,
        color: 'from-violet-500 to-purple-600',
        borderColor: 'border-violet-500/40',
        glowColor: 'shadow-violet-500/20',
        accentBg: 'bg-violet-500/10',
        accentText: 'text-violet-400',
        models: [
            { name: 'VLRM (BLIP-2)', detail: 'Generates prompt-independent captions for each background ROI' },
            { name: 'CLIP / SigLIP Encoder', detail: 'Encodes ROI images + captions, computes Mahalanobis distance against class centroids' },
        ],
        inputs: ['Background ROIs from Stage 1', 'Full image context'],
        outputs: ['Novel object labels with confidence scores', 'Caption + image similarity scores fused via α-weighted combination'],
        details: [
            'VLRM generates a free-form caption per cropped ROI — no prompt needed',
            'CLIP encodes both the ROI image and caption independently',
            'Mahalanobis distance (not cosine similarity) is used against class centroids with shared covariance matrix',
            'Score fusion: final_score = α × caption_score + (1 − α) × image_score',
        ],
    },
    {
        id: 3,
        title: 'Stage 3: Refinement',
        icon: Layers,
        color: 'from-amber-500 to-orange-600',
        borderColor: 'border-amber-500/40',
        glowColor: 'shadow-amber-500/20',
        accentBg: 'bg-amber-500/10',
        accentText: 'text-amber-400',
        models: [
            { name: 'Hybrid Matching (OT + Hungarian)', detail: 'Two-branch matching — Optimal Transport with dynamic top-k + Hungarian one-to-one' },
            { name: 'SAM (ViT-H)', detail: 'Refines segmentation masks for all detected objects' },
            { name: 'SRM (Score Reliability Module)', detail: 'MinMax normalizes detector + SAM scores, then multiplies element-wise' },
        ],
        inputs: ['Known detections from Stage 1', 'Novel detections from Stage 2'],
        outputs: ['Final refined detections with masks', 'Calibrated confidence scores', 'NMS-filtered results'],
        details: [
            'Branch 1: Optimal Transport with dynamic top-k for flexible many-to-one matching',
            'Branch 2: Hungarian algorithm for strict one-to-one matching',
            'Matched detections get confidence boost; unmatched get penalty',
            'SAM generates precise segmentation masks for all final detections',
            'SRM: MinMax(detector_score) × MinMax(SAM_mask_score) → final confidence',
            'NMS removes duplicate detections as the final step',
        ],
    },
]

export default function PipelinePage() {
    const [expanded, setExpanded] = useState(null)

    const toggle = (id) => setExpanded(prev => prev === id ? null : id)

    return (
        <div className="relative overflow-hidden">
            {/* Background */}
            <div className="absolute top-20 right-1/4 w-72 h-72 bg-violet-500/8 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute bottom-20 left-1/4 w-80 h-80 bg-amber-500/5 rounded-full blur-3xl pointer-events-none" />

            <section className="max-w-4xl mx-auto px-4 sm:px-6 pt-12 pb-24">
                <motion.div initial="hidden" animate="visible">
                    {/* Header */}
                    <motion.div className="text-center mb-14" variants={fadeUp}>
                        <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-4">
                            Pipeline{' '}
                            <span className="bg-gradient-to-r from-primary-400 to-violet-400 bg-clip-text text-transparent">
                                Visualization
                            </span>
                        </h1>
                        <p className="text-surface-400 text-lg max-w-2xl mx-auto">
                            Explore the 3-stage cooperative pipeline. Click each stage to see detailed model information, inputs, and outputs.
                        </p>
                    </motion.div>

                    {/* Pipeline Stages */}
                    <div className="space-y-4">
                        {stages.map((stage, idx) => (
                            <motion.div key={stage.id} variants={fadeUp} custom={idx + 1}>
                                {/* Connector */}
                                {idx > 0 && (
                                    <div className="flex justify-center -mt-1 mb-3">
                                        <motion.div
                                            className="flex flex-col items-center"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: idx * 0.2 + 0.3 }}
                                        >
                                            <div className="w-0.5 h-6 bg-gradient-to-b from-surface-600 to-surface-700" />
                                            <ArrowDown className="w-4 h-4 text-surface-600 -mt-1" />
                                        </motion.div>
                                    </div>
                                )}

                                {/* Stage Card */}
                                <div
                                    className={`glass-card overflow-hidden cursor-pointer transition-all duration-300 ${expanded === stage.id
                                        ? `${stage.borderColor} shadow-lg ${stage.glowColor}`
                                        : 'hover:border-surface-600'
                                        }`}
                                    onClick={() => toggle(stage.id)}
                                >
                                    {/* Header */}
                                    <div className="p-5 sm:p-6 flex items-center gap-4">
                                        <div className={`flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center shadow-lg`}>
                                            <stage.icon className="w-6 h-6 text-white" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <h3 className="text-white font-bold text-lg">{stage.title}</h3>
                                            <p className="text-surface-500 text-xs mt-0.5">
                                                {stage.models.map(m => m.name).join(' · ')}
                                            </p>
                                        </div>
                                        <div className={`flex-shrink-0 w-8 h-8 rounded-lg ${stage.accentBg} flex items-center justify-center transition-transform duration-300 ${expanded === stage.id ? 'rotate-180' : ''}`}>
                                            <ChevronDown className={`w-4 h-4 ${stage.accentText}`} />
                                        </div>
                                    </div>

                                    {/* Expandable Content */}
                                    <AnimatePresence>
                                        {expanded === stage.id && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: 'auto', opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                transition={{ duration: 0.3, ease: 'easeInOut' }}
                                                className="overflow-hidden"
                                            >
                                                <div className="px-5 sm:px-6 pb-6 pt-2 border-t border-surface-700/50">
                                                    {/* Models */}
                                                    <div className="mb-5">
                                                        <h4 className={`text-xs font-bold uppercase tracking-wider ${stage.accentText} mb-3`}>
                                                            Models
                                                        </h4>
                                                        <div className="space-y-2">
                                                            {stage.models.map(model => (
                                                                <div key={model.name} className={`${stage.accentBg} rounded-lg p-3`}>
                                                                    <p className="text-white font-medium text-sm">{model.name}</p>
                                                                    <p className="text-surface-400 text-xs mt-0.5">{model.detail}</p>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>

                                                    {/* I/O */}
                                                    <div className="grid sm:grid-cols-2 gap-4 mb-5">
                                                        <div>
                                                            <h4 className="text-xs font-bold uppercase tracking-wider text-surface-500 mb-2">Input</h4>
                                                            <ul className="space-y-1">
                                                                {stage.inputs.map((item, i) => (
                                                                    <li key={i} className="text-surface-300 text-sm flex items-start gap-2">
                                                                        <span className="text-primary-400 mt-1">›</span> {item}
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                        <div>
                                                            <h4 className="text-xs font-bold uppercase tracking-wider text-surface-500 mb-2">Output</h4>
                                                            <ul className="space-y-1">
                                                                {stage.outputs.map((item, i) => (
                                                                    <li key={i} className="text-surface-300 text-sm flex items-start gap-2">
                                                                        <span className="text-emerald-400 mt-1">›</span> {item}
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    </div>

                                                    {/* Details */}
                                                    <div>
                                                        <h4 className="text-xs font-bold uppercase tracking-wider text-surface-500 mb-2">How It Works</h4>
                                                        <ul className="space-y-1.5">
                                                            {stage.details.map((detail, i) => (
                                                                <li key={i} className="text-surface-400 text-sm flex items-start gap-2">
                                                                    <div className={`w-1.5 h-1.5 rounded-full ${stage.accentBg} mt-1.5 flex-shrink-0`} />
                                                                    {detail}
                                                                </li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* CTA */}
                    <motion.div className="text-center mt-14" variants={fadeUp} custom={5}>
                        <motion.a
                            href="#detect"
                            className="inline-flex items-center gap-3 bg-gradient-to-r from-primary-500 to-violet-500 hover:from-primary-400 hover:to-violet-400 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 shadow-xl shadow-primary-500/20 hover:shadow-primary-500/35 text-lg"
                            whileHover={{ scale: 1.04 }}
                            whileTap={{ scale: 0.97 }}
                        >
                            <Sparkles className="w-5 h-5" />
                            Start Detection
                            <ArrowRight className="w-5 h-5" />
                        </motion.a>
                    </motion.div>
                </motion.div>
            </section>
        </div>
    )
}
