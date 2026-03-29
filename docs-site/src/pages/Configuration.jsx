export default function Configuration() {
  return (
    <div className="page">
      <h1>Configuration</h1>
      <p className="page-intro">
        All runtime settings are stored in <code>config.json</code>. The GUI reads and writes
        this file automatically. You can also edit it manually.
      </p>

      <h2>Language &amp; STT</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>language</code></td><td>string</td><td><code>"fi"</code></td><td>STT/TTS language code</td></tr>
            <tr><td><code>whisper_model</code></td><td>string</td><td><code>"base"</code></td><td>Whisper model size: <code>tiny</code>, <code>base</code>, <code>small</code>, <code>medium</code>, <code>large-v3</code></td></tr>
          </tbody>
        </table>
      </div>

      <h2>LLM Settings</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Field</th><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>llm_model_path</code></td><td>string</td><td>Path to the GGUF model file</td></tr>
            <tr><td><code>llm_temperature</code></td><td>float</td><td>Sampling temperature for generation</td></tr>
            <tr><td><code>llm_n_ctx</code></td><td>int</td><td>Context window size (tokens)</td></tr>
            <tr><td><code>llm_max_tokens</code></td><td>int</td><td>Maximum tokens per response</td></tr>
            <tr><td><code>max_conversation_turns</code></td><td>int</td><td>Number of turns to keep in history before trimming</td></tr>
          </tbody>
        </table>
      </div>

      <h2>TTS Settings</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Field</th><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>tts_model</code></td><td>string</td><td>Coqui TTS model identifier (from model zoo)</td></tr>
            <tr><td><code>tts_model_path</code></td><td>string</td><td>Path to a local TTS model file (alternative to model zoo)</td></tr>
            <tr><td><code>tts_config_path</code></td><td>string</td><td>Path to the TTS model config (used with local models)</td></tr>
          </tbody>
        </table>
      </div>

      <h2>VAD Settings</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Field</th><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>vad_threshold</code></td><td>float</td><td>Confidence threshold for speech detection (0.0&ndash;1.0)</td></tr>
            <tr><td><code>min_silence_duration_ms</code></td><td>int</td><td>Milliseconds of silence to mark end of utterance</td></tr>
            <tr><td><code>vad_pre_buffer_ms</code></td><td>int</td><td>Milliseconds of audio to keep before detected speech onset</td></tr>
          </tbody>
        </table>
      </div>

      <h2>Environment Variables</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>CUDA_PATH</code></td><td><code>D:\cuda</code></td><td>Path to CUDA toolkit DLLs</td></tr>
          </tbody>
        </table>
      </div>

      <div className="callout">
        <h4>Tip</h4>
        <p>
          The easiest way to configure the chatbot is through the GUI settings panel.
          Changes are saved to <code>config.json</code> automatically and take effect
          on the next pipeline restart.
        </p>
      </div>
    </div>
  )
}
