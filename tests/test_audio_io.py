import numpy as np

from voice_chatbot.config import Config


def _build_audio_io_module(fresh_import, module_factory):
    state = {"play_calls": [], "wait_calls": 0, "streams": []}

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False
            state["streams"].append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    def fake_play(audio, samplerate):
        state["play_calls"].append((audio.copy(), samplerate))

    def fake_wait():
        state["wait_calls"] += 1

    sounddevice = module_factory(
        "sounddevice",
        InputStream=FakeInputStream,
        play=fake_play,
        wait=fake_wait,
    )
    module = fresh_import(
        "voice_chatbot.audio_io",
        stub_modules={"sounddevice": sounddevice},
        clear_modules=["voice_chatbot.audio_io"],
    )
    return module, state


def test_start_and_stop_capture_manage_input_stream(fresh_import, module_factory):
    module, state = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config(sample_rate=22050, channels=2, chunk_samples=256))

    audio.start_capture()
    stream = state["streams"][0]

    assert stream.started is True
    assert stream.kwargs["samplerate"] == 22050
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["blocksize"] == 256

    audio.stop_capture()

    assert stream.stopped is True
    assert stream.closed is True
    assert audio._stream is None


def test_audio_callback_enqueues_copied_first_channel(fresh_import, module_factory):
    module, _ = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config())
    indata = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int16)

    audio._audio_callback(indata, frames=3, time_info=None, status=None)
    queued = audio.get_audio_chunk(timeout=0.0)
    indata[0, 0] = 999

    assert np.array_equal(queued, np.array([1, 2, 3], dtype=np.int16))


def test_get_audio_chunk_returns_none_on_timeout(fresh_import, module_factory):
    module, _ = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config())

    assert audio.get_audio_chunk(timeout=0.0) is None


def test_clear_queue_drains_buffered_audio(fresh_import, module_factory):
    module, _ = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config())

    audio._audio_queue.put(np.array([1, 2], dtype=np.int16))
    audio._audio_queue.put(np.array([3, 4], dtype=np.int16))
    audio.clear_queue()

    assert audio.get_audio_chunk(timeout=0.0) is None


def test_play_audio_uses_sounddevice_play_and_wait(fresh_import, module_factory):
    module, state = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config())
    sample = np.array([0.1, -0.2], dtype=np.float32)

    audio.play_audio(sample, sample_rate=24000)

    assert state["wait_calls"] == 1
    assert len(state["play_calls"]) == 1
    played_audio, samplerate = state["play_calls"][0]
    assert samplerate == 24000
    assert np.array_equal(played_audio, sample)


def test_close_stops_active_stream(fresh_import, module_factory):
    module, state = _build_audio_io_module(fresh_import, module_factory)
    audio = module.AudioIO(Config())
    audio.start_capture()

    audio.close()

    stream = state["streams"][0]
    assert stream.stopped is True
    assert stream.closed is True
