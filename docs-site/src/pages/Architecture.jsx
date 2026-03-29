import CodeBlock from '../components/CodeBlock'

export default function Architecture() {
  return (
    <div className="page">
      <h1>Architecture</h1>
      <p className="page-intro">
        The voice chatbot uses a modular pipeline architecture where each stage can run
        standalone or as a distributed ROS 2 node.
      </p>

      <h2>Pipeline</h2>
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
      </div>

      <h2>Entry Points</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>File</th><th>Purpose</th></tr>
          </thead>
          <tbody>
            <tr><td><code>app.py</code></td><td>PySide6 desktop GUI &mdash; standalone (loads all models in-process)</td></tr>
            <tr><td><code>ros_app.py</code></td><td>PySide6 desktop GUI &mdash; ROS 2 (connects to split nodes via topics)</td></tr>
            <tr><td><code>chatbot.py</code></td><td>Headless CLI runner (single-threaded audio loop)</td></tr>
            <tr><td><code>setup_models.py</code></td><td>Downloads and validates all models before first run</td></tr>
          </tbody>
        </table>
      </div>

      <h2>Pipeline Modules</h2>

      <h3>config.py &mdash; Configuration</h3>
      <p>
        A Python <code>dataclass</code> with JSON persistence. The GUI reads and writes
        <code>config.json</code> at runtime. All pipeline components receive their settings
        from this central configuration.
      </p>

      <h3>audio_io.py &mdash; Audio I/O</h3>
      <p>
        Wraps <code>sounddevice</code> for microphone capture (callback-based input) and
        speaker playback (blocking output). Manages the audio queue that feeds into VAD.
      </p>

      <h3>vad.py &mdash; Voice Activity Detection</h3>
      <p>
        Silero-VAD wrapper with energy gating and a configurable pre-buffer. Detects
        speech onset and offset, producing audio chunks for the STT stage. Key parameters:
      </p>
      <ul>
        <li><code>vad_threshold</code> &mdash; confidence threshold for speech detection</li>
        <li><code>min_silence_duration_ms</code> &mdash; how long silence must persist to end an utterance</li>
        <li><code>vad_pre_buffer_ms</code> &mdash; audio buffered before speech onset (captures word beginnings)</li>
      </ul>

      <h3>stt.py &mdash; Speech-to-Text</h3>
      <p>
        Uses <code>faster-whisper</code> (CTranslate2 backend) for Whisper inference.
        Supports multiple model sizes from <code>tiny</code> to <code>large-v3</code>.
        CUDA-accelerated when available.
      </p>

      <h3>llm.py &mdash; LLM Chat</h3>
      <p>
        Multi-turn chat via <code>llama-cpp-python</code> with GGUF model files.
        Supports CUDA offloading via <code>n_gpu_layers</code>. Maintains conversation
        history with configurable turn trimming.
      </p>

      <h3>tts_engine.py &mdash; Text-to-Speech</h3>
      <p>
        Coqui TTS wrapper supporting both local model files and the TTS model zoo.
        Requires <code>espeak-ng</code> for phonemisation. Outputs audio data that
        is played back through <code>audio_io</code>.
      </p>

      <h2>Self-Trigger Prevention</h2>
      <p>
        A critical design concern: the microphone must not pick up TTS playback and
        re-trigger the pipeline. After TTS playback completes:
      </p>
      <ol>
        <li>The audio input queue is cleared (<code>clear_queue()</code>)</li>
        <li>The VAD state is reset (<code>vad.reset()</code>)</li>
      </ol>
      <p>This ensures the system doesn't enter a feedback loop.</p>

      <h2>Threading Model</h2>
      <div className="callout">
        <h4>GUI Mode (app.py)</h4>
        <p>
          Heavy model loading and inference run in <code>QThread</code> workers with
          <code>QueuedConnection</code> signals back to the main thread. This keeps
          the Qt event loop responsive.
        </p>
      </div>
      <div className="callout">
        <h4>ROS 2 Mode</h4>
        <p>
          Each node uses a queue + daemon thread pattern. ROS callbacks enqueue work,
          and a background thread processes it sequentially.
        </p>
      </div>

      <h2>Design Conventions</h2>
      <ul>
        <li><strong>Finnish-first</strong> &mdash; UI labels, status messages, and the default system prompt are in Finnish</li>
        <li><strong>CUDA DLL setup</strong> &mdash; must run before any CUDA-dependent imports</li>
        <li><strong>Deferred imports</strong> &mdash; heavy libraries are imported inside worker threads to keep GUI startup instant</li>
        <li><strong>Models directory</strong> &mdash; <code>models/</code> is gitignored, created by <code>setup_models.py</code></li>
      </ul>
    </div>
  )
}
