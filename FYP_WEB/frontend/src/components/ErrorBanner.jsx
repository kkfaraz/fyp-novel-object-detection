import { AlertCircle, RefreshCw, X } from 'lucide-react'

export default function ErrorBanner({ message, onRetry, onDismiss }) {
    return (
        <div className="mt-3 bg-rose-500/10 border border-rose-500/30 rounded-lg p-3 animate-slide-down">
            <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                    <p className="text-rose-200 text-xs">{message}</p>
                    {onRetry && (
                        <button
                            onClick={onRetry}
                            className="mt-2 inline-flex items-center gap-1.5 text-rose-300 hover:text-rose-200 text-xs font-medium transition-colors"
                        >
                            <RefreshCw className="w-3 h-3" />
                            Retry
                        </button>
                    )}
                </div>
                {onDismiss && (
                    <button
                        onClick={onDismiss}
                        className="text-rose-400 hover:text-rose-300 transition-colors flex-shrink-0"
                    >
                        <X className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>
        </div>
    )
}
