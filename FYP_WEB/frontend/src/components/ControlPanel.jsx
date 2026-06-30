import { motion } from 'framer-motion'
import { SlidersHorizontal, Eye, EyeOff, Layers, Hash, Bug } from 'lucide-react'

export default function ControlPanel({
    showKnown, setShowKnown,
    showUnknown, setShowUnknown,
    showMasks, setShowMasks,
    showScores, setShowScores,
    confidenceThreshold, setConfidenceThreshold,
    debugMode, setDebugMode,
}) {
    return (
        <motion.div
            className="glass-card p-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
        >
            <h3 className="text-white font-semibold text-xs mb-4 flex items-center gap-2">
                <SlidersHorizontal className="w-3.5 h-3.5 text-primary-400" />
                Controls
            </h3>

            {/* Confidence Slider */}
            <div className="mb-4">
                <div className="flex items-center justify-between mb-1.5">
                    <label className="text-surface-400 text-xs">Confidence Threshold</label>
                    <span className="text-white text-xs font-mono">{(confidenceThreshold * 100).toFixed(0)}%</span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={confidenceThreshold * 100}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                    className="w-full h-1.5 bg-surface-700 rounded-full appearance-none cursor-pointer accent-primary-500"
                />
            </div>

            {/* Toggles */}
            <div className="space-y-2">
                <ToggleRow
                    label="Show Known"
                    icon={<Eye className="w-3.5 h-3.5" />}
                    active={showKnown}
                    onToggle={() => setShowKnown(!showKnown)}
                    color="emerald"
                />
                <ToggleRow
                    label="Show Unknown"
                    icon={<Eye className="w-3.5 h-3.5" />}
                    active={showUnknown}
                    onToggle={() => setShowUnknown(!showUnknown)}
                    color="amber"
                />
                <ToggleRow
                    label="Show Masks"
                    icon={<Layers className="w-3.5 h-3.5" />}
                    active={showMasks}
                    onToggle={() => setShowMasks(!showMasks)}
                    color="violet"
                />
                <ToggleRow
                    label="Show Scores"
                    icon={<Hash className="w-3.5 h-3.5" />}
                    active={showScores}
                    onToggle={() => setShowScores(!showScores)}
                    color="primary"
                />

                {/* Debug Mode */}
                <div className="pt-2 border-t border-surface-700/50">
                    <ToggleRow
                        label="Debug Mode"
                        icon={<Bug className="w-3.5 h-3.5" />}
                        active={debugMode}
                        onToggle={() => setDebugMode(!debugMode)}
                        color="rose"
                    />
                </div>
            </div>
        </motion.div>
    )
}

function ToggleRow({ label, icon, active, onToggle, color }) {
    const colorClasses = {
        emerald: { on: 'bg-emerald-500', off: 'bg-surface-600' },
        amber: { on: 'bg-amber-500', off: 'bg-surface-600' },
        violet: { on: 'bg-violet-500', off: 'bg-surface-600' },
        primary: { on: 'bg-primary-500', off: 'bg-surface-600' },
        rose: { on: 'bg-rose-500', off: 'bg-surface-600' },
    }

    const c = colorClasses[color] || colorClasses.primary

    return (
        <button
            onClick={onToggle}
            className="w-full flex items-center justify-between py-1.5 group"
        >
            <div className="flex items-center gap-2 text-surface-400 group-hover:text-surface-300 transition-colors">
                {icon}
                <span className="text-xs">{label}</span>
            </div>
            <div className={`w-8 h-4 rounded-full transition-colors duration-200 relative ${active ? c.on : c.off}`}>
                <div className={`w-3 h-3 bg-white rounded-full absolute top-0.5 transition-transform duration-200 ${active ? 'translate-x-4.5' : 'translate-x-0.5'}`} />
            </div>
        </button>
    )
}
