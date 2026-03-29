import CodeBlock from '../components/CodeBlock'

export default function QuickStart() {
  return (
    <div className="page">
      <h1>Quick Start</h1>
      <p className="page-intro">
        Get the voice chatbot running on your machine in a few minutes.
      </p>

      <h2>Prerequisites</h2>
      <ul>
        <li><strong>Python 3.11</strong> (managed automatically by pixi)</li>
        <li><strong>CUDA-capable GPU</strong> for real-time inference (recommended)</li>
        <li><strong>pixi</strong> package manager &mdash; <a href="https://pixi.sh" target="_blank" rel="noopener noreferrer">install pixi</a></li>
        <li><strong>espeak-ng</strong> &mdash; required for Coqui TTS phonemisation
          <ul>
            <li>Windows: Install to <code>C:\Program Files\eSpeak NG</code></li>
            <li>Linux: <code>sudo apt install espeak-ng</code></li>
          </ul>
        </li>
        <li><strong>CUDA toolkit</strong> &mdash; DLL path defaults to <code>D:\cuda</code> (override with <code>CUDA_PATH</code> env var)</li>
      </ul>

      <h2>1. Install Dependencies</h2>
      <CodeBlock language="bash">{`# Create pixi environment with all conda dependencies
pixi install

# Install pip packages (torch, TTS, faster-whisper, etc.)
pixi run install-python-deps`}</CodeBlock>

      <h2>2. Download Models</h2>
      <CodeBlock language="bash">{`# Download LLM, Whisper, VAD, and TTS model files
pixi run setup-models`}</CodeBlock>
      <p>
        Models are downloaded to the <code>models/</code> directory (gitignored).
        This includes Whisper, Silero-VAD, the GGUF LLM, and Coqui TTS models.
      </p>

      <h2>3. Run the Application</h2>

      <h3>GUI (Primary Interface)</h3>
      <CodeBlock language="bash">{`pixi run app
# or directly:
python app.py`}</CodeBlock>
      <p>
        The PySide6 desktop GUI provides real-time status display, waveform visualization,
        conversation history, and a settings panel.
      </p>

      <h3>CLI (Headless)</h3>
      <CodeBlock language="bash">{`python chatbot.py`}</CodeBlock>
      <p>
        The CLI runner operates in a single-threaded audio loop &mdash; useful for
        headless servers or debugging.
      </p>

      <h2>Platform Notes</h2>

      <div className="callout">
        <h4>Windows</h4>
        <p>
          Primary development platform. CUDA DLLs are loaded before any GPU imports.
          The install script targets CUDA 12.8 by default.
        </p>
      </div>

      <div className="callout">
        <h4>Linux</h4>
        <p>
          Fully supported via the Linux install script. Uses generic CUDA detection.
          Make sure <code>espeak-ng</code> is installed system-wide.
        </p>
      </div>
    </div>
  )
}
