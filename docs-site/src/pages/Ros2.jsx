import CodeBlock from '../components/CodeBlock'

export default function Ros2() {
  return (
    <div className="page">
      <h1>ROS 2 Integration</h1>
      <p className="page-intro">
        The chatbot can run as distributed ROS 2 Humble nodes via pixi and Robostack,
        enabling modular deployment across processes or machines.
      </p>

      <h2>Overview</h2>
      <p>
        The ROS 2 integration splits the pipeline into three independent nodes that
        communicate via ROS topics. This allows each component (STT, LLM, TTS) to
        run in its own process with independent resource management.
      </p>

      <h2>Build</h2>
      <CodeBlock language="bash">{`# Build the ROS 2 package with colcon (run once)
pixi run build`}</CodeBlock>

      <h2>Running Nodes</h2>

      <h3>Option A: All Four Tabs</h3>
      <p>Opens STT, LLM, TTS, and GUI in separate Windows Terminal tabs:</p>
      <CodeBlock language="bash">{`pixi run ros-start`}</CodeBlock>

      <h3>Option B: Individual Nodes</h3>
      <p>Run each node in a separate terminal (build once first):</p>
      <CodeBlock language="bash">{`pixi run ros-stt    # STT node: mic + VAD + Whisper
pixi run ros-llm    # LLM node: LLaMA chat inference
pixi run ros-tts    # TTS node: Coqui TTS + audio playback
pixi run ros-app    # PySide6 GUI (connects to running nodes)`}</CodeBlock>

      <h3>Option C: Launch File</h3>
      <p>Run all three processing nodes in one process group:</p>
      <CodeBlock language="bash">{`pixi run ros-launch`}</CodeBlock>

      <h3>Legacy Monolithic Node</h3>
      <p>Loads everything sequentially in a single node (not recommended for production):</p>
      <CodeBlock language="bash">{`pixi run ros-run`}</CodeBlock>

      <h2>Node Architecture</h2>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Node</th><th>File</th><th>Responsibilities</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>STT Node</strong></td>
              <td><code>voice_chatbot_ros/stt_node.py</code></td>
              <td>Microphone capture, VAD processing, Whisper transcription. Publishes recognized text.</td>
            </tr>
            <tr>
              <td><strong>LLM Node</strong></td>
              <td><code>voice_chatbot_ros/llm_node.py</code></td>
              <td>Receives transcribed text, runs LLaMA inference, publishes response text.</td>
            </tr>
            <tr>
              <td><strong>TTS Node</strong></td>
              <td><code>voice_chatbot_ros/tts_node.py</code></td>
              <td>Receives response text, synthesizes speech with Coqui TTS, plays audio.</td>
            </tr>
            <tr>
              <td><strong>GUI Node</strong></td>
              <td><code>ros_app.py</code></td>
              <td>PySide6 interface that subscribes to all topics for visualization and control.</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Launch File</h2>
      <p>
        The launch file at <code>launch/voice_chatbot.launch.py</code> starts all three
        processing nodes (STT, LLM, TTS) in a single process group. The GUI node
        (<code>ros_app.py</code>) is run separately to keep the Qt event loop independent.
      </p>

      <h2>Build Tools</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>File</th><th>Purpose</th></tr>
          </thead>
          <tbody>
            <tr><td><code>setup.py</code></td><td>ROS 2 package setup (colcon / ament_python)</td></tr>
            <tr><td><code>package.xml</code></td><td>ROS 2 package manifest</td></tr>
            <tr><td><code>tools/ros_start_all.bat/.sh</code></td><td>Launch all 4 tabs (STT, LLM, TTS, GUI)</td></tr>
            <tr><td><code>tools/ros_run_node_pixi.bat/.sh</code></td><td>Run a single ROS 2 node via pixi</td></tr>
            <tr><td><code>tools/ros_launch_pixi.bat/.sh</code></td><td>Run the launch file via pixi</td></tr>
            <tr><td><code>tools/ensure_setuptools_compat.py</code></td><td>Enforce setuptools 69.5&ndash;79.x for colcon</td></tr>
          </tbody>
        </table>
      </div>

      <div className="callout">
        <h4>Note</h4>
        <p>
          PySide6 (pip-installed) can conflict with Robostack's Qt packages.
          The app includes a DLL workaround in <code>app.py</code> to handle this.
          See the source code for details.
        </p>
      </div>
    </div>
  )
}
