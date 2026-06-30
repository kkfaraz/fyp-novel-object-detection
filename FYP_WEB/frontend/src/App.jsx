import Navbar from './components/Navbar'
import IntroductionPage from './pages/IntroductionPage'
import PipelinePage from './pages/PipelinePage'
import DetectionPage from './pages/DetectionPage'

export default function App() {
  return (
    <div className="min-h-screen bg-surface-900">
      <Navbar />
      <main>
        <section id="intro">
          <IntroductionPage />
        </section>
        <section id="pipeline">
          <PipelinePage />
        </section>
        <section id="detect">
          <DetectionPage />
        </section>
      </main>
    </div>
  )
}
