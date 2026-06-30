import { motion } from 'framer-motion'
import { Loader2, Search, Tag, Layers } from 'lucide-react'

const stages = {
    stage1: {
        label: 'Stage 1: Detection',
        description: 'GroundingDINO + Mask R-CNN running...',
        icon: Search,
        color: 'from-cyan-500 to-blue-600',
        progress: 33,
    },
    stage2: {
        label: 'Stage 2: Unknown Labelling',
        description: 'VLRM captioning + CLIP scoring...',
        icon: Tag,
        color: 'from-violet-500 to-purple-600',
        progress: 66,
    },
    stage3: {
        label: 'Stage 3: Refinement',
        description: 'Hybrid Matching + SAM + SRM...',
        icon: Layers,
        color: 'from-amber-500 to-orange-600',
        progress: 90,
    },
}

export default function LoadingOverlay({ currentStage }) {
    const stage = stages[currentStage] || stages.stage1
    const Icon = stage.icon

    return (
        <motion.div
            className="glass-card p-8 text-center"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
        >
            {/* Animated spinner */}
            <div className="relative w-20 h-20 mx-auto mb-6">
                <div className={`absolute inset-0 rounded-full bg-gradient-to-r ${stage.color} opacity-20 animate-ping`} />
                <div className={`absolute inset-0 rounded-full bg-gradient-to-r ${stage.color} opacity-10`} />
                <div className="absolute inset-2 rounded-full bg-surface-900 flex items-center justify-center">
                    <Icon className="w-8 h-8 text-white animate-pulse" />
                </div>
                <Loader2 className="absolute inset-0 w-20 h-20 text-primary-400 animate-spin opacity-40" />
            </div>

            <h3 className="text-white font-bold text-lg mb-1">{stage.label}</h3>
            <p className="text-surface-400 text-sm mb-6">{stage.description}</p>

            {/* Progress Bar */}
            <div className="w-full max-w-xs mx-auto">
                <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                    <motion.div
                        className={`h-full bg-gradient-to-r ${stage.color} rounded-full`}
                        initial={{ width: '5%' }}
                        animate={{ width: `${stage.progress}%` }}
                        transition={{ duration: 2, ease: 'easeInOut' }}
                    />
                </div>
                <div className="flex justify-between mt-2">
                    {Object.entries(stages).map(([key, s]) => (
                        <span
                            key={key}
                            className={`text-[10px] font-medium ${key === currentStage ? 'text-white' : 'text-surface-600'
                                }`}
                        >
                            S{key.slice(-1)}
                        </span>
                    ))}
                </div>
            </div>

            <p className="text-surface-600 text-xs mt-4">This may take a few seconds...</p>
        </motion.div>
    )
}
