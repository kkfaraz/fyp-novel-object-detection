import { useEffect, useState } from 'react'
import { Microscope, Workflow, ScanSearch } from 'lucide-react'

const navItems = [
    { path: '#intro', label: 'Introduction', icon: Microscope },
    { path: '#pipeline', label: 'Pipeline', icon: Workflow },
    { path: '#detect', label: 'Detection', icon: ScanSearch },
]

export default function Navbar() {
    const [activeHash, setActiveHash] = useState('#intro')

    useEffect(() => {
        const handleScroll = () => {
            const sections = navItems.map(item => document.querySelector(item.path));
            let current = '#intro';
            for (const section of sections) {
                if (section) {
                    const rect = section.getBoundingClientRect();
                    // If the section's top is past the middle of the viewport
                    if (rect.top <= window.innerHeight / 2) {
                        current = '#' + section.id;
                    }
                }
            }
            setActiveHash(current);
        };

        window.addEventListener('scroll', handleScroll);
        // Initial check
        handleScroll();
        return () => window.removeEventListener('scroll', handleScroll);
    }, [])

    return (
        <nav className="sticky top-0 z-50 backdrop-blur-xl bg-surface-900/80 border-b border-surface-700/50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <a href="#intro" className="flex items-center gap-2 group">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/25 group-hover:shadow-primary-500/40 transition-shadow">
                            <Microscope className="w-4 h-4 text-white" />
                        </div>
                        <span className="text-white font-bold text-lg hidden sm:inline">
                            Co<span className="text-primary-400">NOD</span>
                        </span>
                    </a>

                    {/* Nav Links */}
                    <div className="flex items-center gap-1">
                        {navItems.map(({ path, label, icon: Icon }) => {
                            const isActive = activeHash === path
                            return (
                                <a
                                    key={path}
                                    href={path}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${isActive
                                        ? 'bg-primary-500/15 text-primary-300 shadow-inner'
                                        : 'text-surface-400 hover:text-white hover:bg-surface-800/50'
                                        }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="hidden sm:inline">{label}</span>
                                </a>
                            )
                        })}
                    </div>
                </div>
            </div>
        </nav>
    )
}
