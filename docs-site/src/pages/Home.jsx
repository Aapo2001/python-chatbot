import { Link } from 'react-router-dom'

const features = [
  {
    title: 'Voice Activity Detection',
    desc: 'Silero-VAD with energy gating and pre-buffering for accurate speech detection.',
    icon: '~',
  },
  {
    title: 'Speech-to-Text',
    desc: 'CTranslate2-based Whisper via faster-whisper with CUDA acceleration.',
    icon: '>',
  },
  {
    title: 'LLM Chat',
    desc: 'Multi-turn conversations powered by LLaMA GGUF models via llama-cpp-python.',
    icon: '#',
  },
  {
    title: 'Text-to-Speech',
    desc: 'Natural voice synthesis with Coqui TTS and espeak-ng phonemisation.',
    icon: ')',
  },
  {
    title: 'Desktop GUI',
    desc: 'PySide6 (Qt 6) interface with real-time status, waveform display, and settings.',
    icon: '[',
  },
  {
    title: 'ROS 2 Integration',
    desc: 'Split-node architecture on ROS 2 Humble via Robostack for modular deployment.',
    icon: '{',
  },
]

export default function Home() {
  return (
    <div className="page">
      <div className="hero-section">
        <h1>Voice Chatbot</h1>
        <p className="hero-subtitle">
          A local speech-to-speech voice assistant with PySide6 GUI, CLI, and ROS 2 Humble integration.
          Fully offline, GPU-accelerated, and privacy-first.
        </p>
        <div className="hero-actions">
          <Link to="/quick-start" className="btn btn-primary">Get Started</Link>
          <Link to="/architecture" className="btn btn-secondary">Learn More</Link>
        </div>
      </div>

      <div className="pipeline-section">
        <h2>Pipeline Overview</h2>
        <div className="pipeline-diagram">
          <div className="pipeline-flow">
            <span className="pipeline-node">Microphone</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">AudioIO</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">VAD</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">STT</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">LLM</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">TTS</span>
            <span className="pipeline-arrow" />
            <span className="pipeline-node">Speaker</span>
          </div>
          <p className="pipeline-note">
            After playback, the audio queue is cleared and VAD is reset to prevent self-triggering.
          </p>
        </div>
      </div>

      <h2>Features</h2>
      <div className="feature-grid">
        {features.map(f => (
          <div key={f.title} className="feature-card">
            <div className="feature-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>

      <div className="tech-stack">
        <h2>Tech Stack</h2>
        <div className="tech-grid">
          {[
            ['Python 3.11', 'Runtime'],
            ['PySide6', 'Qt 6 GUI'],
            ['silero-vad', 'Voice Detection'],
            ['faster-whisper', 'Speech-to-Text'],
            ['llama-cpp-python', 'LLM Inference'],
            ['Coqui TTS', 'Text-to-Speech'],
            ['sounddevice', 'Audio I/O'],
            ['ROS 2 Humble', 'Robot Framework'],
            ['pixi', 'Package Manager'],
            ['CUDA', 'GPU Acceleration'],
          ].map(([name, role]) => (
            <div key={name} className="tech-item">
              <strong>{name}</strong>
              <span>{role}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
