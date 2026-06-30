import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRight, Brain, Eye, Layers, Sparkles, Zap, Shield, Search, Tag, Box, X, Play } from 'lucide-react'

const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: (i = 0) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.1, duration: 0.6, ease: 'easeOut' },
    }),
}

const stageData = [
    {
        stage: 1,
        title: 'Initialization',
        subtitle: 'Detection',
        icon: Search,
        color: 'from-cyan-500 to-blue-600',
        glow: 'shadow-cyan-500/30',
        description: 'GroundingDINO performs open-vocabulary detection while Mask R-CNN generates class-agnostic proposals.',
        models: ['GroundingDINO (Swin-T)', 'Mask R-CNN (R101-FPN)'],
    },
    {
        stage: 2,
        title: 'Unknown Labelling',
        subtitle: 'Semantic Reasoning',
        icon: Tag,
        color: 'from-violet-500 to-purple-600',
        glow: 'shadow-violet-500/30',
        description: 'VLRM generates captions for ROIs. CLIP computes Mahalanobis distance for novel category scoring.',
        models: ['VLRM (BLIP-2)', 'CLIP / SigLIP Encoder'],
    },
    {
        stage: 3,
        title: 'Refinement',
        subtitle: 'Score & Mask Enhancement',
        icon: Layers,
        color: 'from-amber-500 to-orange-600',
        glow: 'shadow-amber-500/30',
        description: 'Hybrid Matching fuses detections. SAM refines masks. SRM normalizes and combines final scores.',
        models: ['SAM (ViT-H)', 'Hybrid OT + Hungarian', 'SRM Module'],
    },
]

const concepts = [
    {
        title: 'Closed-Set Detection',
        icon: Shield,
        description: 'Can only detect object categories seen during training. Limited to a fixed set of classes.',
        color: 'text-rose-400',
        border: 'border-rose-500/30',
        bg: 'bg-rose-500/10',
    },
    {
        title: 'Open-Vocabulary Detection',
        icon: Eye,
        description: 'Detects objects described by text prompts. Extends beyond training categories via language–vision models.',
        color: 'text-blue-400',
        border: 'border-blue-500/30',
        bg: 'bg-blue-500/10',
    },
    {
        title: 'Novel Object Detection',
        icon: Sparkles,
        description: 'Discovers completely unknown objects without any text prompt or prior knowledge — zero-training required.',
        color: 'text-emerald-400',
        border: 'border-emerald-500/30',
        bg: 'bg-emerald-500/10',
    },
]

export default function IntroductionPage() {
    const [showVideoModal, setShowVideoModal] = useState(false)

    return (
        <div className="relative overflow-hidden">
            {/* Background gradient orbs */}
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-violet-500/10 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl pointer-events-none" />

            {/* Hero Section */}
            <section className="relative max-w-6xl mx-auto px-4 sm:px-6 pt-16 pb-20">
                <motion.div
                    className="text-center"
                    initial="hidden"
                    animate="visible"
                    variants={fadeUp}
                >
                    {/* Badge */}
                    <motion.div
                        className="inline-flex items-center gap-2 bg-gradient-to-r from-primary-500/15 to-violet-500/15 px-5 py-2 rounded-full mb-6 border border-primary-500/30"
                        variants={fadeUp}
                        custom={0}
                    >
                        <Zap className="w-4 h-4 text-primary-400" />
                        <span className="text-primary-300 text-sm font-medium">Zero-Training Pipeline</span>
                    </motion.div>

                    {/* Title */}
                    <motion.h1
                        className="text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-tight mb-6"
                        variants={fadeUp}
                        custom={1}
                    >
                        <span className="bg-gradient-to-r from-white via-primary-200 to-primary-400 bg-clip-text text-transparent">
                            Cooperative Novel
                        </span>
                        <br />
                        <span className="bg-gradient-to-r from-primary-400 via-violet-400 to-amber-400 bg-clip-text text-transparent animate-gradient">
                            Object Detection
                        </span>
                    </motion.h1>

                    {/* Subtitle */}
                    <motion.p
                        className="text-lg sm:text-xl text-surface-400 max-w-2xl mx-auto mb-10 leading-relaxed"
                        variants={fadeUp}
                        custom={2}
                    >
                        Discover unknown objects in images using cooperative foundation models —
                        no training required. Powered by GroundingDINO, CLIP, SAM, and VLRM working together.
                    </motion.p>

                    {/* CTA Button */}
                    <motion.button
                        onClick={() => setShowVideoModal(true)}
                        className="inline-flex items-center gap-3 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-400 hover:to-primary-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 shadow-xl shadow-primary-500/25 hover:shadow-primary-500/40 hover:scale-105 text-lg"
                        variants={fadeUp}
                        custom={3}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        Launch Demo
                        <ArrowRight className="w-5 h-5" />
                    </motion.button>
                </motion.div>
            </section>

            {/* What is Novel Object Detection? */}
            <section className="relative max-w-6xl mx-auto px-4 sm:px-6 pb-20">
                <motion.div
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: '-50px' }}
                >
                    <motion.h2
                        className="text-3xl sm:text-4xl font-bold text-center text-white mb-4"
                        variants={fadeUp}
                    >
                        What is Novel Object Detection?
                    </motion.h2>
                    <motion.p
                        className="text-surface-400 text-center max-w-3xl mx-auto mb-12 text-lg"
                        variants={fadeUp}
                        custom={1}
                    >
                        Traditional detectors fail on objects they weren't trained on. Novel Object Detection
                        (NOD) aims to identify <em>both</em> known and completely unknown objects in a scene.
                    </motion.p>

                    <div className="grid md:grid-cols-3 gap-6">
                        {concepts.map((concept, idx) => (
                            <motion.div
                                key={concept.title}
                                className={`glass-card glass-card-hover p-6 ${concept.border}`}
                                variants={fadeUp}
                                custom={idx + 2}
                            >
                                <div className={`w-12 h-12 rounded-xl ${concept.bg} flex items-center justify-center mb-4`}>
                                    <concept.icon className={`w-6 h-6 ${concept.color}`} />
                                </div>
                                <h3 className="text-white font-semibold text-lg mb-2">{concept.title}</h3>
                                <p className="text-surface-400 text-sm leading-relaxed">{concept.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </section>

            {/* 3-Stage Pipeline Diagram */}
            <section className="relative max-w-6xl mx-auto px-4 sm:px-6 pb-24">
                <motion.div
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: '-50px' }}
                >
                    <motion.h2
                        className="text-3xl sm:text-4xl font-bold text-center text-white mb-4"
                        variants={fadeUp}
                    >
                        3-Stage Cooperative Pipeline
                    </motion.h2>
                    <motion.p
                        className="text-surface-400 text-center max-w-3xl mx-auto mb-14 text-lg"
                        variants={fadeUp}
                        custom={1}
                    >
                        Foundation models cooperate across three stages — no fine-tuning needed.
                    </motion.p>

                    <div className="space-y-6">
                        {stageData.map((stage, idx) => (
                            <motion.div
                                key={stage.stage}
                                variants={fadeUp}
                                custom={idx + 2}
                            >
                                {/* Connector arrow */}
                                {idx > 0 && (
                                    <div className="flex justify-center -mt-2 mb-4">
                                        <div className="w-0.5 h-8 bg-gradient-to-b from-surface-600 to-surface-700 rounded-full" />
                                    </div>
                                )}

                                <div className={`glass-card p-6 sm:p-8 relative overflow-hidden group hover:border-primary-500/30 transition-all duration-300`}>
                                    {/* Glow accent */}
                                    <div className={`absolute inset-0 bg-gradient-to-r ${stage.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />

                                    <div className="relative flex flex-col sm:flex-row items-start gap-5">
                                        {/* Stage number + icon */}
                                        <div className={`flex-shrink-0 w-14 h-14 rounded-2xl bg-gradient-to-br ${stage.color} flex items-center justify-center shadow-lg ${stage.glow}`}>
                                            <stage.icon className="w-7 h-7 text-white" />
                                        </div>

                                        <div className="flex-1">
                                            <div className="flex flex-wrap items-center gap-3 mb-2">
                                                <span className="text-surface-500 text-xs font-bold uppercase tracking-wider">Stage {stage.stage}</span>
                                                <h3 className="text-white font-bold text-xl">{stage.title}</h3>
                                                <span className="text-surface-500 text-xs">— {stage.subtitle}</span>
                                            </div>
                                            <p className="text-surface-400 text-sm leading-relaxed mb-3">{stage.description}</p>
                                            <div className="flex flex-wrap gap-2">
                                                {stage.models.map(model => (
                                                    <span
                                                        key={model}
                                                        className="text-xs font-medium px-3 py-1 rounded-full bg-surface-800/80 text-surface-300 border border-surface-700"
                                                    >
                                                        {model}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>

                {/* Foundation Models Grid */}
                <motion.div
                    className="mt-20"
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true }}
                >
                    <motion.h2
                        className="text-3xl font-bold text-center text-white mb-12"
                        variants={fadeUp}
                    >
                        Foundation Models Used
                    </motion.h2>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {[
                            { name: 'GroundingDINO', role: 'Open-Vocab Detection' },
                            { name: 'Mask R-CNN', role: 'Region Proposals' },
                            { name: 'VLRM (BLIP-2)', role: 'Region Captioning' },
                            { name: 'CLIP / SigLIP', role: 'Vision-Language Scoring' },
                            { name: 'SAM ViT-H', role: 'Mask Refinement' },
                        ].map((model, idx) => (
                            <motion.div
                                key={model.name}
                                className="glass-card glass-card-hover p-4 text-center"
                                variants={fadeUp}
                                custom={idx}
                            >
                                <div className="w-10 h-10 rounded-full bg-primary-500/15 flex items-center justify-center mx-auto mb-3">
                                    <Brain className="w-5 h-5 text-primary-400" />
                                </div>
                                <p className="text-white font-semibold text-sm mb-1">{model.name}</p>
                                <p className="text-surface-500 text-xs">{model.role}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </section>

            {/* Video Demo Modal */}
            {showVideoModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    {/* Backdrop */}
                    <motion.div
                        className="absolute inset-0 bg-black/80 backdrop-blur-md"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        onClick={() => setShowVideoModal(false)}
                    />
                    
                    {/* Modal Content Container */}
                    <motion.div
                        className="relative bg-surface-900 border border-surface-700/50 rounded-2xl overflow-hidden max-w-4xl w-full shadow-2xl z-10"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-4 border-b border-surface-800">
                            <h3 className="text-xl font-bold text-white flex items-center gap-2">
                                <Play className="w-5 h-5 text-primary-400 fill-primary-400/20" />
                                Demo: How to Detect Objects
                            </h3>
                            <button
                                onClick={() => setShowVideoModal(false)}
                                className="text-surface-400 hover:text-white hover:bg-surface-800 p-2 rounded-lg transition-colors"
                            >
                                <X className="w-6 h-6" />
                            </button>
                        </div>
                        
                        {/* Video body */}
                        <div className="relative aspect-video bg-black flex items-center justify-center">
                            <video
                                src="/demo.webm"
                                controls
                                autoPlay
                                className="w-full h-full"
                            />
                        </div>
                        
                        {/* Footer */}
                        <div className="p-4 bg-surface-950 flex justify-between items-center gap-4">
                            <p className="text-sm text-surface-400 hidden sm:block">
                                Learn how to upload an image and view multi-stage detection.
                            </p>
                            <a
                                href="#detect"
                                onClick={() => setShowVideoModal(false)}
                                className="inline-flex items-center gap-2 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-400 hover:to-primary-500 text-white font-semibold py-2 px-5 rounded-lg transition-all shadow-md text-sm ml-auto"
                            >
                                Try It Yourself
                                <ArrowRight className="w-4 h-4" />
                            </a>
                        </div>
                    </motion.div>
                </div>
            )}
        </div>
    )
}
