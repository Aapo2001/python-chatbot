export default function ApiReference() {
  return (
    <div className="page">
      <h1>API Reference</h1>
      <p className="page-intro">
        Module-level reference for the voice chatbot pipeline components.
      </p>

      <h2>config.py</h2>
      <div className="api-section">
        <h3><code>Config</code></h3>
        <p>
          A <code>@dataclass</code> holding all runtime settings. Supports JSON serialization
          for persistence.
        </p>
        <div className="api-methods">
          <div className="api-method">
            <code>Config.load(path: str = "config.json") -&gt; Config</code>
            <p>Load configuration from a JSON file. Returns defaults if file doesn't exist.</p>
          </div>
          <div className="api-method">
            <code>Config.save(path: str = "config.json") -&gt; None</code>
            <p>Persist the current configuration to a JSON file.</p>
          </div>
        </div>
      </div>

      <h2>audio_io.py</h2>
      <div className="api-section">
        <h3><code>AudioIO</code></h3>
        <p>Manages microphone capture and speaker playback via <code>sounddevice</code>.</p>
        <div className="api-methods">
          <div className="api-method">
            <code>AudioIO.start() -&gt; None</code>
            <p>Begin capturing audio from the microphone.</p>
          </div>
          <div className="api-method">
            <code>AudioIO.stop() -&gt; None</code>
            <p>Stop audio capture.</p>
          </div>
          <div className="api-method">
            <code>AudioIO.play(audio_data: np.ndarray, sample_rate: int) -&gt; None</code>
            <p>Play audio data through the speaker (blocking).</p>
          </div>
          <div className="api-method">
            <code>AudioIO.clear_queue() -&gt; None</code>
            <p>Clear the audio input queue. Used after TTS playback to prevent self-triggering.</p>
          </div>
        </div>
      </div>

      <h2>vad.py</h2>
      <div className="api-section">
        <h3><code>VoiceActivityDetector</code></h3>
        <p>Silero-VAD wrapper with energy gating and pre-buffering.</p>
        <div className="api-methods">
          <div className="api-method">
            <code>VoiceActivityDetector(config: Config)</code>
            <p>Initialize with VAD threshold, silence duration, and pre-buffer settings from config.</p>
          </div>
          <div className="api-method">
            <code>VoiceActivityDetector.process(audio_chunk: np.ndarray) -&gt; Optional[np.ndarray]</code>
            <p>Feed an audio chunk. Returns the complete utterance when speech ends, or <code>None</code> if still listening.</p>
          </div>
          <div className="api-method">
            <code>VoiceActivityDetector.reset() -&gt; None</code>
            <p>Reset internal state. Called after TTS playback.</p>
          </div>
        </div>
      </div>

      <h2>stt.py</h2>
      <div className="api-section">
        <h3><code>SpeechToText</code></h3>
        <p>Whisper-based transcription via <code>faster-whisper</code>.</p>
        <div className="api-methods">
          <div className="api-method">
            <code>SpeechToText(config: Config)</code>
            <p>Initialize with model size and language from config. Loads the CTranslate2 model.</p>
          </div>
          <div className="api-method">
            <code>SpeechToText.transcribe(audio: np.ndarray) -&gt; str</code>
            <p>Transcribe an audio array to text.</p>
          </div>
        </div>
      </div>

      <h2>llm.py</h2>
      <div className="api-section">
        <h3><code>ChatLLM</code></h3>
        <p>Multi-turn chat via <code>llama-cpp-python</code> with GGUF models.</p>
        <div className="api-methods">
          <div className="api-method">
            <code>ChatLLM(config: Config)</code>
            <p>Initialize with model path, context size, and generation parameters from config.</p>
          </div>
          <div className="api-method">
            <code>ChatLLM.chat(user_message: str) -&gt; str</code>
            <p>Send a message and get the assistant's response. Maintains conversation history.</p>
          </div>
          <div className="api-method">
            <code>ChatLLM.reset() -&gt; None</code>
            <p>Clear conversation history.</p>
          </div>
        </div>
      </div>

      <h2>tts_engine.py</h2>
      <div className="api-section">
        <h3><code>TextToSpeech</code></h3>
        <p>Coqui TTS wrapper for speech synthesis.</p>
        <div className="api-methods">
          <div className="api-method">
            <code>TextToSpeech(config: Config)</code>
            <p>Initialize with TTS model settings from config. Requires <code>espeak-ng</code>.</p>
          </div>
          <div className="api-method">
            <code>TextToSpeech.synthesize(text: str) -&gt; tuple[np.ndarray, int]</code>
            <p>Convert text to audio. Returns <code>(audio_data, sample_rate)</code>.</p>
          </div>
        </div>
      </div>

      <div className="callout">
        <h4>Note</h4>
        <p>
          This API reference shows the public interface of each module. The actual
          method signatures may include additional optional parameters. Refer to the
          source code for complete details.
        </p>
      </div>
    </div>
  )
}
