import base64
import importlib
import json
import os
import shutil
import sys
import threading
import warnings
from datetime import datetime
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf


# ---------------------------------------------------------------------------
# NumPy 2.x compatibility shim for Applio/RVC (and other legacy dependencies)
# ---------------------------------------------------------------------------
# NumPy 1.24 removed the deprecated scalar aliases np.bool, np.int, np.float,
# np.complex, np.object, np.str, np.long and np.unicode. Applio/RVC still
# references some of them (e.g. ``np.long`` and ``np.ulong``), which raises
# AttributeError. NumPy 2.0 additionally removed/renamed np.float_,
# np.complex_, np.string_, np.unicode_, np.NaN, np.Inf, etc. Restore them so
# legacy code keeps working.
def _patch_numpy_compat():
    """Restore NumPy aliases removed in 1.24 / 2.0 for legacy code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Aliases removed in NumPy 1.24
        legacy_aliases = {
            "bool": bool,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "str": str,
            "long": np.int_,
            "ulong": np.dtype("L").type,
            "unicode": str,
        }
        for alias, target in legacy_aliases.items():
            try:
                getattr(np, alias)
            except AttributeError:
                setattr(np, alias, target)

        # Scalar aliases removed/renamed in NumPy 2.0
        renamed = {
            "float_": "float64",
            "complex_": "complex128",
            "longfloat": "longdouble",
            "singlecomplex": "complex64",
            "cfloat": "complex128",
            "longcomplex": "clongdouble",
            "clongfloat": "clongdouble",
            "string_": "bytes_",
            "unicode_": "str_",
        }
        for alias, target_name in renamed.items():
            if not hasattr(np, alias) and hasattr(np, target_name):
                setattr(np, alias, getattr(np, target_name))

        # Constants removed in NumPy 2.0
        constants = {
            "Inf": np.inf,
            "Infinity": np.inf,
            "infty": np.inf,
            "NaN": np.nan,
            "PINF": np.inf,
            "NINF": -np.inf,
        }
        for alias, value in constants.items():
            if not hasattr(np, alias):
                setattr(np, alias, value)

        # RankWarning: moved to numpy.exceptions in 1.25, REMOVED in NumPy 2.0.
        # Applio's infer.py does ``from numpy.exceptions import RankWarning``,
        # so the exception module itself must expose it again.
        try:
            import numpy.exceptions as np_exceptions

            if not hasattr(np_exceptions, "RankWarning"):
                rank_warning = type("RankWarning", (Warning,), {
                    "__module__": "numpy.exceptions",
                    "__doc__": "Polyfit may be poorly conditioned.",
                })
                np_exceptions.RankWarning = rank_warning
                setattr(np, "RankWarning", rank_warning)
        except Exception:
            pass


# Apply immediately so any later import (Applio/RVC, torchcrepe, faiss, ...)
# already sees the restored aliases.
_patch_numpy_compat()

# Suppress the noisy NumPy/SciPy version mismatch warning emitted by numba/scipy
# when a legacy Applio env (NumPy 1.26.x) is paired with a newer SciPy build.
warnings.filterwarnings(
    "ignore",
    message=".*NumPy version.*required for this version of SciPy.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Gradio / Matplotlib backend crash guard
# ---------------------------------------------------------------------------
# Gradio wraps every API call in its Matplotlib backend manager. Some Gradio
# versions can fail to initialize the manager if Matplotlib is broken or
# partially imported, then raise again while cleaning up. Use a tolerant
# implementation so API calls and timer ticks are not interrupted.
def _patch_gradio_matplotlib_backend():
    """Replace Gradio's fragile Matplotlib backend manager with a safe version."""
    try:
        from gradio import utils as gradio_utils
    except Exception:
        return

    if getattr(gradio_utils, "_super_impersonator_mpl_patched", False):
        return

    class _SafeMatplotlibBackendManager:
        """Context manager that never raises, even with a broken matplotlib."""

        def __enter__(self):
            self._original_backend = None
            try:
                import matplotlib

                self._original_backend = matplotlib.get_backend()
                matplotlib.use("agg")
            except Exception:
                self._original_backend = None
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._original_backend is None:
                return False
            try:
                import matplotlib

                matplotlib.use(self._original_backend)
            except Exception:
                pass
            return False

    gradio_utils.MatplotlibBackendMananger = _SafeMatplotlibBackendManager
    gradio_utils._super_impersonator_mpl_patched = True

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass


_patch_gradio_matplotlib_backend()


# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------

ENGINE_DIR = Path(__file__).resolve().parent
APPLIO_DIR = Path(
    os.environ.get("APPLIO_DIR", r"C:\Users\User\Downloads\ApplioV3.7.1")
).expanduser()
FFMPEG_BIN_DIR = ENGINE_DIR
MODELS_DIR = ENGINE_DIR / "assets" / "models"
APPLIO_MODELS_DIR = APPLIO_DIR / "rvc" / "infer"
RECORDINGS_DIR = ENGINE_DIR / "assets" / "recordings"
TEMP_DIR = ENGINE_DIR / "assets" / "temp"
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", r"C:\Backups"))
AVATAR_DIR = Path(
    os.environ.get(
        "AVATAR_DIR",
        r"C:\Users\User\Downloads\Super Impersonator app by don24crk\Engine\avatare",
    )
)

# Virtualenv holding torch, torchcrepe, faiss, etc.
#
# Default: the Applio installation's own .venv. Override with the
# APPLIO_ENV_DIR environment variable if your environment lives elsewhere.
DEFAULT_ENV_DIR = Path(r"C:\Users\User\Downloads\ApplioV3.7.1\env")

# Fallback candidates, tried in order when the configured dir does not exist.
ENV_DIR_CANDIDATES = list(
    dict.fromkeys(
        [
            DEFAULT_ENV_DIR,
            APPLIO_DIR / "env",
            APPLIO_DIR / ".venv",
            APPLIO_DIR / "venv",
            ENGINE_DIR / ".venv",
            ENGINE_DIR / "env",
        ]
    )
)


def _resolve_env_dir():
    """Return the configured env dir, or the first existing fallback."""
    configured = os.environ.get("APPLIO_ENV_DIR")
    if configured:
        return Path(configured).expanduser()
    for candidate in ENV_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return DEFAULT_ENV_DIR


APPLIO_ENV_DIR = _resolve_env_dir()

ffmpeg_path = FFMPEG_BIN_DIR / "ffmpeg.exe"
if ffmpeg_path.exists():
    os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")


# Bundled Applio core files and their targets inside the Applio installation
APPLIO_CORE_FILES = {
    "infer.py": APPLIO_DIR / "rvc" / "infer" / "infer.py",
    "pipeline.py": APPLIO_DIR / "rvc" / "infer" / "pipeline.py",
    "worker.py": APPLIO_DIR / "rvc" / "realtime" / "worker.py",
}


def sync_applio_core_files():
    """Copy bundled Applio core overrides into the Applio installation."""
    for filename, target in APPLIO_CORE_FILES.items():
        source = ENGINE_DIR / filename
        if not source.is_file():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(target.name + ".bak")
                if not backup.exists():
                    shutil.copy2(target, backup)
            if not target.exists() or source.read_bytes() != target.read_bytes():
                shutil.copy2(source, target)
                print(f"Applio-Datei aktualisiert: {target}")
        except Exception as exc:
            print(f"Warnung: Konnte {filename} nicht nach {target.parent} kopieren: {exc}")


def _iter_env_layouts(env_dir):
    """Yield the import-relevant subdirectories of a venv/env directory."""
    env_dir = Path(env_dir)
    yield env_dir
    yield env_dir / "Lib"
    yield env_dir / "Lib" / "site-packages"
    yield env_dir / "DLLs"
    yield env_dir / "Scripts"
    yield env_dir / "bin"
    yield env_dir / "lib"
    for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for site_packages in sorted(env_dir.glob(pattern)):
            yield site_packages


def _register_windows_dll_dirs():
    """Let Windows find torch's bundled DLLs."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    env_roots = []
    for env_dir in [APPLIO_ENV_DIR, *ENV_DIR_CANDIDATES]:
        if env_dir.is_dir() and env_dir not in env_roots:
            env_roots.append(env_dir)
    candidates = []
    for env_dir in env_roots:
        candidates.extend(
            [
                env_dir / "Lib" / "site-packages" / "torch" / "lib",
                env_dir / "Lib" / "site-packages" / "torch" / "bin",
                env_dir / "Lib" / "site-packages" / "torch" / "lib" / "torch_rt",
                env_dir / "Lib" / "site-packages" / "torch" / "lib" / "torch_rt" / "dlls",
                env_dir / "Lib" / "site-packages" / "torch" / "lib" / "torch_rt" / "nvidia",
                env_dir / "DLLs",
                env_dir / "Library" / "bin",
            ]
        )
    for directory in candidates:
        if not directory.is_dir():
            continue
        try:
            os.add_dll_directory(str(directory))
        except OSError as exc:
            print(f"DLL-Verzeichnis nicht registriert ({directory}): {exc}")
        current_path = os.environ.get("PATH", "")
        if str(directory) not in current_path.split(os.pathsep):
            os.environ["PATH"] = str(directory) + os.pathsep + current_path


def inject_applio_paths():
    """Add the configured Applio directories (including its env) to Python's path."""
    # Applio's own env ALWAYS wins; the Engine venv is only a fallback.
    # (Injecting it first mixed SciPy-for-NumPy-2.x with Applio's NumPy 1.26.4.)
    paths = [
        APPLIO_DIR,
        APPLIO_DIR / "src",
        APPLIO_DIR / "rvc",
        APPLIO_DIR / "rvc" / "infer",
    ]

    env_dirs = []
    for env_dir in [APPLIO_ENV_DIR, *ENV_DIR_CANDIDATES]:
        if env_dir not in env_dirs:
            env_dirs.append(env_dir)
    for env_dir in env_dirs:
        if env_dir.is_dir():
            paths.extend(_iter_env_layouts(env_dir))

    engine_fallbacks = []
    for root in (APPLIO_DIR, ENGINE_DIR):
        if not root.exists():
            continue
        for pattern in (
            "*/Lib/site-packages",
            "*/lib/python*/site-packages",
            "*/site-packages",
        ):
            for site_packages in sorted(root.glob(pattern)):
                if not site_packages.is_dir() or site_packages in paths:
                    continue
                if str(site_packages).startswith(str(ENGINE_DIR)):
                    engine_fallbacks.append(site_packages)
                else:
                    paths.append(site_packages)
    paths.extend(engine_fallbacks)

    seen = set()
    new_paths = []
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        if path.exists() and path_str not in sys.path:
            new_paths.append(path_str)
    if new_paths:
        sys.path[0:0] = new_paths

    extra_bin_dirs = []
    for env_dir in env_dirs:
        extra_bin_dirs.extend([env_dir / "Scripts", env_dir / "bin", env_dir / "DLLs"])
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    for directory in extra_bin_dirs:
        if directory.is_dir() and str(directory) not in path_parts:
            path_parts.insert(0, str(directory))
    if path_parts:
        os.environ["PATH"] = os.pathsep.join(path_parts)

    _register_windows_dll_dirs()
    os.environ["CUSTOM_APPLIO_PATH"] = str(APPLIO_DIR)
    os.environ.setdefault("APPLIO_ENV_DIR", str(APPLIO_ENV_DIR))


def describe_applio_environment():
    """Human-readable summary of what was found; helps debugging imports."""
    lines = [f"APPLIO_DIR     : {APPLIO_DIR} ({'OK' if APPLIO_DIR.is_dir() else 'FEHLT'})"]
    lines.append(
        f"APPLIO_ENV_DIR : {APPLIO_ENV_DIR} "
        f"({'OK' if APPLIO_ENV_DIR.is_dir() else 'FEHLT'})"
    )
    for label, candidate in (
        ("env/Lib/site-packages", APPLIO_ENV_DIR / "Lib" / "site-packages"),
        ("env/Scripts", APPLIO_ENV_DIR / "Scripts"),
        ("env/DLLs", APPLIO_ENV_DIR / "DLLs"),
        ("env/Lib/site-packages/torch", APPLIO_ENV_DIR / "Lib" / "site-packages" / "torch"),
    ):
        lines.append(f"  {label:<24}: {'OK' if candidate.exists() else 'FEHLT'}")

    lines.append("Env candidates (fallback order):")
    for candidate in ENV_DIR_CANDIDATES:
        marker = "  <- aktiv" if candidate == APPLIO_ENV_DIR else ""
        lines.append(
            f"  {'OK  ' if candidate.is_dir() else 'FEHLT'} {candidate}{marker}"
        )
    return "\n".join(lines)


def load_vc_class():
    """Load the voice-converter class across Applio V3 and legacy RVC layouts."""
    _patch_numpy_compat()
    inject_applio_paths()
    candidates = (
        ("rvc.infer.infer", "VoiceConverter"),
        ("rvc.infer.infer", "VC"),
        ("rvc.infer.modules.vc.modules", "VC"),
        ("infer.modules.vc.modules", "VC"),
        ("rvc.modules.vc.modules", "VC"),
    )
    errors = []

    for module_name, attribute in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        vc_class = getattr(module, attribute, None)
        if vc_class is not None:
            return vc_class
        errors.append(f"{module_name}: {attribute} class not found")

    searched_paths = "\n".join(f"  - {name}.{attr}" for name, attr in candidates)
    details = "\n".join(f"  - {error}" for error in errors)
    raise ImportError(
        "Could not import the Applio/RVC voice converter class. "
        f"Checked:\n{searched_paths}\nImport errors:\n{details}\n"
        f"Configured Applio directory: {APPLIO_DIR}\n"
        f"Environment check:\n{describe_applio_environment()}\n"
        "Hint: the app looks for a virtualenv containing torch/torchcrepe in "
        "this order:\n"
        f"  1. {DEFAULT_ENV_DIR} (default)\n"
        f"  2. {APPLIO_DIR / 'env'}\n"
        f"  3. {APPLIO_DIR / '.venv'}\n"
        f"  4. {APPLIO_DIR / 'venv'}\n"
        f"  5. {ENGINE_DIR / '.venv'}\n"
        "Set the APPLIO_ENV_DIR environment variable to override this."
    )


sync_applio_core_files()
inject_applio_paths()


# ---------------------------------------------------------------------------
# Shared live-audio state
# ---------------------------------------------------------------------------

live_streaming = False
live_stream = None
is_recording = False
recorded_blocks = []
current_live_vu_level = -60.0
recording_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

STUDIO_PRESETS = {
    "Standard (Manuell)": {
        "vocal_gain": 0.0,
        "noise_red": 0,
        "gate_thresh": -50,
        "gate_release": 150,
        "comp_thresh": -20,
        "comp_ratio": 3.5,
        "comp_attack": 15,
        "comp_release": 200,
        "eq_low": 0,
        "eq_mid": 2,
        "eq_high": 1,
        "echo_time": 0.3,
        "echo_fb": 30,
        "reverb_size": 0.5,
        "fx_mix": 15,
    },
    "Radio-Stimme (Fat Voice)": {
        "vocal_gain": 4.0,
        "noise_red": 40,
        "gate_thresh": -45,
        "gate_release": 120,
        "comp_thresh": -24,
        "comp_ratio": 5.0,
        "comp_attack": 10,
        "comp_release": 150,
        "eq_low": 6,
        "eq_mid": -1,
        "eq_high": 4,
        "echo_time": 0.2,
        "echo_fb": 15,
        "reverb_size": 0.3,
        "fx_mix": 20,
    },
    "Podcast (Warm & Nah)": {
        "vocal_gain": 3.0,
        "noise_red": 60,
        "gate_thresh": -48,
        "gate_release": 200,
        "comp_thresh": -18,
        "comp_ratio": 3.0,
        "comp_attack": 20,
        "comp_release": 250,
        "eq_low": 4,
        "eq_mid": 3,
        "eq_high": 0,
        "echo_time": 0.0,
        "echo_fb": 0,
        "reverb_size": 0.1,
        "fx_mix": 5,
    },
    "Clear Vocal (Kristallklar)": {
        "vocal_gain": 2.0,
        "noise_red": 30,
        "gate_thresh": -52,
        "gate_release": 100,
        "comp_thresh": -15,
        "comp_ratio": 2.5,
        "comp_attack": 8,
        "comp_release": 100,
        "eq_low": -3,
        "eq_mid": 1,
        "eq_high": 6,
        "echo_time": 0.4,
        "echo_fb": 25,
        "reverb_size": 0.6,
        "fx_mix": 25,
    },
}

PRESET_OUTPUT_KEYS = [
    "vocal_gain",
    "noise_red",
    "gate_thresh",
    "gate_release",
    "comp_thresh",
    "comp_ratio",
    "comp_attack",
    "comp_release",
    "eq_low",
    "eq_mid",
    "eq_high",
    "echo_time",
    "echo_fb",
    "reverb_size",
    "fx_mix",
]


def load_preset_values(preset_name):
    """Return Gradio updates in the same order as the preset controls."""
    preset = STUDIO_PRESETS.get(preset_name)
    if preset is None:
        return [gr.update() for _ in PRESET_OUTPUT_KEYS]
    return [gr.update(value=preset[key]) for key in PRESET_OUTPUT_KEYS]


# ---------------------------------------------------------------------------
# Audio and model helpers
# ---------------------------------------------------------------------------


def calculate_auto_pitch(audio_path, target_model_path):
    """Estimate a source F0 and return an approximate shift toward the target."""
    if not audio_path or not Path(audio_path).is_file():
        return 0, "Fehler: Keine gültige Audiodatei für Pitch-Analyse geladen."

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1)
        if mono.size < sample_rate // 4:
            return 0, "Pitch-Erkennung nicht möglich: Audiodatei ist zu kurz."

        max_samples = min(mono.size, int(sample_rate * 2.0))
        start = max(0, (mono.size - max_samples) // 2)
        signal = mono[start : start + max_samples]
        signal = signal - np.mean(signal)
        if not np.all(np.isfinite(signal)) or float(np.max(np.abs(signal))) < 1e-5:
            return 0, "Pitch-Erkennung nicht möglich: Signal ist still oder ungültig."

        fft_size = 1 << (2 * signal.size - 1).bit_length()
        spectrum = np.fft.rfft(signal, n=fft_size)
        autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)
        autocorrelation = autocorrelation[: signal.size]

        min_lag = max(1, int(sample_rate / 400.0))
        max_lag = min(signal.size - 1, int(sample_rate / 80.0))
        if max_lag <= min_lag:
            return 0, "Pitch-Erkennung nicht möglich: Audiosegment ist zu kurz."

        segment = autocorrelation[min_lag : max_lag + 1]
        peak_lag = int(np.argmax(segment)) + min_lag
        detected_f0 = float(sample_rate) / peak_lag
        if not np.isfinite(detected_f0) or detected_f0 <= 0:
            return 0, "Pitch-Erkennung ungenau: Keine stabile Grundfrequenz gefunden."

        model_name = Path(target_model_path).stem.lower() if target_model_path else ""
        female_keywords = (
            "miley",
            "kalomira",
            "cyrus",
            "girl",
            "female",
            "woman",
            "sängerin",
        )
        is_target_female = any(keyword in model_name for keyword in female_keywords)
        target_f0 = 240.0 if is_target_female else 130.0

        semitones = int(round(12.0 * np.log2(target_f0 / detected_f0)))
        semitones = max(-24, min(24, semitones))
        gender_text = "Weiblich (Sopran)" if is_target_female else "Männlich (Bariton)"
        message = (
            f"Pitch erkannt: ca. {int(round(detected_f0))} Hz. Ziel-Modell: {gender_text}. "
            f"Automatischer Pitch-Shift: {semitones} Halbtöne."
        )
        return semitones, message
    except Exception as exc:
        return 0, f"Fehler bei Pitch-Auto-Detection: {exc}. Nutze Standardwert (0)."


def get_audio_devices():
    """Return device labels and valid default input/output labels."""
    try:
        devices = sd.query_devices()
        input_devices = [
            f"{index}: {device['name']}"
            for index, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ]
        output_devices = [
            f"{index}: {device['name']}"
            for index, device in enumerate(devices)
            if device["max_output_channels"] > 0
        ]
        default_input, default_output = sd.default.device

        def default_label(index, channel_key, choices):
            if index is not None and 0 <= index < len(devices):
                if devices[index][channel_key] > 0:
                    label = f"{index}: {devices[index]['name']}"
                    if label in choices:
                        return label
            return choices[0] if choices else None

        if not input_devices:
            input_devices = ["0: No input device available"]
        if not output_devices:
            output_devices = ["0: No output device available"]

        return (
            input_devices,
            output_devices,
            default_label(default_input, "max_input_channels", input_devices),
            default_label(default_output, "max_output_channels", output_devices),
        )
    except Exception as exc:
        print(f"Fehler beim Laden der Audiogeräte: {exc}")
        fallback_in = ["0: Standard-Eingang"]
        fallback_out = ["0: Standard-Ausgang"]
        return fallback_in, fallback_out, fallback_in[0], fallback_out[0]


def parse_device_index(device_label):
    try:
        return int(str(device_label).split(":", 1)[0])
    except (ValueError, IndexError, AttributeError):
        return None


def scan_rvc_models():
    """Find relative .pth paths beneath either supported model directory."""
    models = set()
    for directory in (MODELS_DIR, APPLIO_MODELS_DIR):
        if not directory.exists():
            continue
        for model_path in directory.rglob("*.pth"):
            try:
                models.add(model_path.relative_to(directory).as_posix())
            except ValueError:
                continue
    return sorted(models)


def find_associated_index(model_rel_path):
    if not model_rel_path:
        return ""
    model_base = Path(model_rel_path).stem.lower()
    for directory in (MODELS_DIR, APPLIO_MODELS_DIR):
        if not directory.exists():
            continue
        for index_path in directory.rglob("*.index"):
            if model_base in index_path.stem.lower():
                return str(index_path)
    return ""


def resolve_model_path(model_rel_path):
    """Resolve a selected model safely within one of the configured roots."""
    if not model_rel_path:
        return None
    relative_path = Path(model_rel_path)
    if relative_path.is_absolute():
        return None

    for directory in (MODELS_DIR, APPLIO_MODELS_DIR):
        root = directory.resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def make_output_path(directory, prefix, extension):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{timestamp}.{extension}"


def convert_codec(input_wav, target_codec, mp3_bitrate=320, output_dir=None):
    """Export audio through pydub/FFmpeg, fallback to WAV copy on fail."""
    target_codec = str(target_codec).lower()
    if target_codec not in {"wav", "mp3", "flac", "wma"}:
        raise ValueError(f"Unsupported export format: {target_codec}")

    input_path = Path(input_wav)
    if not input_path.is_file():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {input_path}")

    directory = Path(output_dir) if output_dir else ENGINE_DIR
    prefix = "live_rec" if output_dir else "master_bounce"
    final_path = make_output_path(directory, prefix, target_codec)

    try:
        if target_codec == "wav":
            shutil.copy2(input_path, final_path)
        else:
            from pydub import AudioSegment

            if ffmpeg_path.exists():
                AudioSegment.converter = str(ffmpeg_path)
            audio = AudioSegment.from_file(str(input_path))
            export_options = {}
            if target_codec == "mp3":
                export_options["bitrate"] = f"{int(mp3_bitrate)}k"
            audio.export(str(final_path), format=target_codec, **export_options)
    except Exception as exc:
        print(f"Codec-Fehler ({exc}); nutze WAV-Fallback.")
        final_path = make_output_path(directory, prefix, "wav")
        shutil.copy2(input_path, final_path)

    if output_dir:
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, BACKUP_DIR / final_path.name)
        except Exception as exc:
            print(f"Redundanz-Backup fehlgeschlagen: {exc}")
    return str(final_path)


# ---------------------------------------------------------------------------
# Studio effects and sidechain ducking
# ---------------------------------------------------------------------------


def _resample_audio(audio, source_rate, target_rate):
    """Resample channels using scipy when sample rates differ."""
    if source_rate == target_rate:
        return audio
    try:
        from scipy.signal import resample_poly

        divisor = int(np.gcd(int(source_rate), int(target_rate)))
        return resample_poly(
            audio,
            int(target_rate) // divisor,
            int(source_rate) // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    except ImportError as exc:
        raise RuntimeError("scipy ist erforderlich, um die BGM-Samplerate anzupassen.") from exc


def mix_background_music(
    vocal_path,
    bgm_path,
    bgm_vol_db,
    ducking_db,
    attack_ms,
    release_ms,
    target_sr,
):
    """Mix background music under vocals, using vocal-driven ducking."""
    try:
        vocal, vocal_sr = sf.read(vocal_path, dtype="float32", always_2d=True)
        bgm, bgm_sr = sf.read(bgm_path, dtype="float32", always_2d=True)
        target_sr = int(target_sr)
        if vocal_sr != target_sr:
            vocal = _resample_audio(vocal, vocal_sr, target_sr)
        if bgm_sr != target_sr:
            bgm = _resample_audio(bgm, bgm_sr, target_sr)

        def stereo(audio):
            if audio.shape[1] == 1:
                return np.repeat(audio, 2, axis=1)
            if audio.shape[1] > 2:
                return audio[:, :2]
            return audio

        vocal = stereo(vocal)
        bgm = stereo(bgm)
        vocal_length = len(vocal)
        if vocal_length == 0:
            raise RuntimeError("Die Vocal-Datei enthält keine Audiodaten.")
        if len(bgm) < vocal_length:
            bgm = np.pad(bgm, ((0, vocal_length - len(bgm)), (0, 0)))
        else:
            bgm = bgm[:vocal_length]

        bgm *= 10.0 ** (float(bgm_vol_db) / 20.0)
        duck_gain = 10.0 ** (-abs(float(ducking_db)) / 20.0)
        vocal_level = np.mean(np.abs(vocal), axis=1)

        frame_size = max(1, int(target_sr * 0.01))
        frame_count = int(np.ceil(vocal_length / frame_size))
        frame_levels = np.zeros(frame_count, dtype=np.float32)
        for frame in range(frame_count):
            start = frame * frame_size
            frame_levels[frame] = float(np.max(vocal_level[start : start + frame_size]))

        attack_seconds = max(0.001, float(attack_ms) / 1000.0)
        release_seconds = max(0.001, float(release_ms) / 1000.0)
        attack_coefficient = np.exp(-frame_size / (target_sr * attack_seconds))
        release_coefficient = np.exp(-frame_size / (target_sr * release_seconds))
        envelope_frames = np.ones(frame_count, dtype=np.float32)
        current_gain = 1.0
        for frame, level in enumerate(frame_levels):
            target_gain = duck_gain if level > 0.01 else 1.0
            coefficient = (
                attack_coefficient if target_gain < current_gain else release_coefficient
            )
            current_gain = coefficient * current_gain + (1.0 - coefficient) * target_gain
            envelope_frames[frame] = current_gain

        envelope = np.repeat(envelope_frames, frame_size)[:vocal_length]
        mixed = np.clip(vocal + bgm * envelope[:, None], -1.0, 1.0)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        mixed_path = make_output_path(TEMP_DIR, "master_mixed", "wav")
        sf.write(str(mixed_path), mixed, target_sr)
        return str(mixed_path)
    except Exception as exc:
        print(f"Fehler beim Sidechain-Mixing: {exc}")
        return vocal_path


def build_effect_boards(
    vocal_gain,
    gate_thresh,
    gate_release,
    comp_thresh,
    comp_ratio,
    comp_attack,
    comp_release,
    eq_low,
    eq_mid,
    eq_high,
    echo_time,
    echo_feedback,
    reverb_size,
):
    from pedalboard import (
        Compressor,
        Delay,
        Gain,
        HighShelfFilter,
        LowShelfFilter,
        NoiseGate,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    dry_board = Pedalboard(
        [
            Gain(gain_db=float(vocal_gain)),
            NoiseGate(threshold_db=float(gate_thresh), release_ms=float(gate_release)),
            Compressor(
                threshold_db=float(comp_thresh),
                ratio=float(comp_ratio),
                attack_ms=float(comp_attack),
                release_ms=float(comp_release),
            ),
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=float(eq_low)),
            PeakFilter(cutoff_frequency_hz=2500.0, gain_db=float(eq_mid), q=1.0),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=float(eq_high)),
        ]
    )
    wet_board = Pedalboard(
        [
            Delay(
                delay_seconds=float(echo_time),
                feedback=float(echo_feedback) / 100.0,
                mix=1.0,
            ),
            Reverb(room_size=float(reverb_size), wet_level=1.0, dry_level=0.0),
        ]
    )
    return dry_board, wet_board


def apply_studio_fx(
    input_wav_path,
    vocal_gain,
    noise_red,
    gate_thresh,
    gate_release,
    comp_thresh,
    comp_ratio,
    comp_attack,
    comp_release,
    eq_low,
    eq_mid,
    eq_high,
    echo_time,
    echo_fb,
    reverb_size,
    fx_mix,
    target_sr,
):
    del noise_red, target_sr
    try:
        dry_board, wet_board = build_effect_boards(
            vocal_gain,
            gate_thresh,
            gate_release,
            comp_thresh,
            comp_ratio,
            comp_attack,
            comp_release,
            eq_low,
            eq_mid,
            eq_high,
            echo_time,
            echo_fb,
            reverb_size,
        )
        audio, source_sr = sf.read(input_wav_path, dtype="float32", always_2d=True)
        audio = audio.T
        dry = dry_board(audio, source_sr)
        wet = wet_board(dry, source_sr)
        mix = float(np.clip(float(fx_mix) / 100.0, 0.0, 1.0))
        output = np.clip(dry * (1.0 - mix) + wet * mix, -1.0, 1.0).T
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_path = make_output_path(TEMP_DIR, "temp_master", "wav")
        sf.write(str(output_path), output, source_sr)
        return str(output_path)
    except Exception as exc:
        print(f"FX-Chain-Fehler: {exc}")
        return input_wav_path


def normalize_rvc_result(result, sample_rate):
    """Normalize Applio/RVC return shapes to a WAV filepath."""
    if isinstance(result, (str, os.PathLike)) and Path(result).is_file():
        return str(result)

    if isinstance(result, tuple) and len(result) == 2:
        if isinstance(result[1], tuple) and len(result[1]) == 2:
            result = result[1]
        elif not isinstance(result[0], (int, float, np.integer, np.floating)):
            raise RuntimeError(f"RVC inference returned an error: {result[0]}")

    if isinstance(result, tuple) and len(result) == 2:
        returned_sr, audio = result
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]
        elif audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        if audio.ndim != 2 or audio.size == 0:
            raise RuntimeError("RVC returned empty or invalid audio data.")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        output_path = make_output_path(TEMP_DIR, "rvc_output", "wav")
        sf.write(str(output_path), audio, int(returned_sr or sample_rate))
        return str(output_path)

    raise RuntimeError(f"Unrecognized RVC audio result: {type(result).__name__}")


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------


def process_audio(
    input_audio,
    rvc_model_path,
    vocal_pitch,
    f0_method,
    index_rate,
    protect,
    filter_radius,
    rms_mix_rate,
    hop_length,
    vocal_gain,
    noise_red,
    gate_threshold,
    gate_release,
    comp_thresh,
    comp_ratio,
    comp_attack,
    comp_release,
    eq_low,
    eq_mid,
    eq_high,
    echo_time,
    echo_feedback,
    reverb_size,
    fx_mix,
    export_codec,
    sample_rate,
    mp3_bitrate,
    bgm_file,
    bgm_volume,
    ducking_intensity,
    ducking_attack,
    ducking_release,
):
    inject_applio_paths()
    input_path = Path(input_audio) if input_audio else ENGINE_DIR / "audio.wav"
    if not input_path.is_file():
        return None, f"Fehler: Audiodatei nicht gefunden: {input_path}"
    if not rvc_model_path:
        return None, "Fehler: Kein RVC-Modell ausgewählt."

    model_path = resolve_model_path(rvc_model_path)
    if model_path is None:
        return None, f"Fehler: Modell nicht gefunden: {rvc_model_path}"

    try:
        target_sr = int("".join(filter(str.isdigit, str(sample_rate))))
        if target_sr <= 0:
            raise ValueError(f"Ungültige Sample-Rate: {sample_rate}")

        VC = load_vc_class()

        vc = VC()
        raw_output_path = make_output_path(TEMP_DIR, "rvc_raw", "wav")
        if hasattr(vc, "convert_audio"):
            vc.convert_audio(
                audio_input_path=str(input_path),
                audio_output_path=str(raw_output_path),
                model_path=str(model_path),
                index_path=find_associated_index(rvc_model_path) or "",
                pitch=int(vocal_pitch),
                f0_method=str(f0_method),
                index_rate=float(index_rate),
                volume_envelope=float(rms_mix_rate),
                protect=float(protect),
                hop_length=(
                    int(hop_length) if "crepe" in str(f0_method).lower() else 128
                ),
                resample_sr=target_sr,
                export_format="WAV",
            )
            rvc_wav_path = str(raw_output_path)
        else:
            vc.get_vc(str(model_path))
            result = vc.vc_single(
                sid=0,
                input_audio_path=str(input_path),
                f0_up_key=int(vocal_pitch),
                f0_file=None,
                f0_method=str(f0_method),
                file_index=find_associated_index(rvc_model_path),
                file_index2="",
                index_rate=float(index_rate),
                filter_radius=int(filter_radius),
                resample_sr=target_sr,
                rms_mix_rate=float(rms_mix_rate),
                protect=float(protect),
                hop_length=(
                    int(hop_length) if "crepe" in str(f0_method).lower() else 128
                ),
            )
            rvc_wav_path = normalize_rvc_result(result, target_sr)
        master_path = apply_studio_fx(
            rvc_wav_path,
            vocal_gain,
            noise_red,
            gate_threshold,
            gate_release,
            comp_thresh,
            comp_ratio,
            comp_attack,
            comp_release,
            eq_low,
            eq_mid,
            eq_high,
            echo_time,
            echo_feedback,
            reverb_size,
            fx_mix,
            target_sr,
        )
        if bgm_file and Path(bgm_file).is_file():
            master_path = mix_background_music(
                master_path,
                bgm_file,
                bgm_volume,
                ducking_intensity,
                ducking_attack,
                ducking_release,
                target_sr,
            )
        final_path = convert_codec(master_path, export_codec, mp3_bitrate)
        return final_path, "Studio-Inferenz erfolgreich abgeschlossen!"
    except Exception as exc:
        return None, f"Fehler in der Inferenz-Pipeline: {exc}"


# ---------------------------------------------------------------------------
# Live recording
# ---------------------------------------------------------------------------


def _live_audio_callback(indata, frames, time_info, status):
    global current_live_vu_level
    del frames, time_info
    if status:
        print(f"Audio-Stream-Hinweis: {status}")
    block = np.asarray(indata, dtype=np.float32).copy()
    if block.ndim == 1:
        block = block[:, None]
    with recording_lock:
        if is_recording:
            recorded_blocks.append(block)
    peak = float(np.max(np.abs(block))) if block.size else 0.0
    current_live_vu_level = max(-60.0, 20.0 * np.log10(peak)) if peak > 1e-6 else -60.0


def update_live_vu():
    if not live_streaming:
        return gr.update(value=-60.0), "OFFLINE"
    level = float(np.clip(current_live_vu_level, -60.0, 0.0))
    return gr.update(value=level), "MONITORING" if level < -1.0 else "CLIPPING"


def start_live_recording(input_device, output_device, block_size):
    global is_recording, recorded_blocks, live_streaming, live_stream
    del output_device
    if live_streaming:
        return "Aufnahme läuft bereits..."

    input_idx = parse_device_index(input_device)
    if input_idx is None:
        return "Fehler: Ungültiges Eingabegerät."
    try:
        device_info = sd.query_devices(input_idx)
        if int(device_info["max_input_channels"]) < 1:
            return "Fehler: Das ausgewählte Gerät hat keinen Audioeingang."
        channels = min(int(device_info["max_input_channels"]), 2)
        samplerate = int(device_info.get("default_samplerate") or 44100)
        with recording_lock:
            recorded_blocks.clear()
            is_recording = True
        live_streaming = True
        live_stream = sd.InputStream(
            device=input_idx,
            channels=channels,
            samplerate=samplerate,
            blocksize=int(block_size),
            dtype="float32",
            callback=_live_audio_callback,
        )
        live_stream.start()
        return "AUFNAHME GESTARTET..."
    except Exception as exc:
        with recording_lock:
            is_recording = False
        live_streaming = False
        live_stream = None
        return f"Fehler beim Starten der Aufnahme: {exc}"


def stop_live_recording(export_codec, mp3_bitrate):
    global is_recording, recorded_blocks, live_streaming, live_stream
    with recording_lock:
        is_recording = False
        blocks = recorded_blocks.copy()
        recorded_blocks.clear()

    live_streaming = False
    used_samplerate = 44100
    if live_stream is not None:
        try:
            used_samplerate = int(live_stream.samplerate)
            live_stream.stop()
            live_stream.close()
        except Exception:
            pass
        finally:
            live_stream = None

    if not blocks:
        return gr.update(value=None), "Keine Audiodaten aufgezeichnet."
    try:
        temp_wav = make_output_path(TEMP_DIR, "live_rec_temp", "wav")
        audio = np.concatenate(blocks, axis=0)
        sf.write(str(temp_wav), audio, used_samplerate)
        final_file = convert_codec(
            str(temp_wav), export_codec, mp3_bitrate, output_dir=RECORDINGS_DIR
        )
        return gr.update(value=final_file), f"Aufnahme exportiert: {Path(final_file).name}"
    except Exception as exc:
        return gr.update(value=None), f"Fehler beim Exportieren: {exc}"


# ---------------------------------------------------------------------------
# Set-card UI helpers
# ---------------------------------------------------------------------------


def get_avatar_base64(model_stem):
    if not AVATAR_DIR.exists():
        return ""
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        image_path = AVATAR_DIR / f"{model_stem}{extension}"
        if image_path.is_file():
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            mime = "jpeg" if extension == ".jpg" else extension[1:]
            return f"data:image/{mime};base64,{encoded}"
    return ""


def generate_setcards_html(models, selected_model):
    if not models:
        return '<div class="setcard-empty">Keine Modelle im Verzeichnis gefunden.</div>'

    cards = []
    for model in models:
        model_stem = Path(model).stem
        is_selected = model == selected_model
        avatar_src = get_avatar_base64(model_stem)
        safe_model = escape(model, quote=True)
        model_json = escape(json.dumps(model), quote=True)
        if avatar_src:
            visual = (
                f'<img class="setcard-avatar" src="{avatar_src}" '
                f'alt="{escape(model_stem, quote=True)}">'
            )
        else:
            visual = (
                f'<div class="setcard-avatar setcard-initial">'
                f'{escape(model_stem[:1].upper())}</div>'
            )
        selected_class = " selected" if is_selected else ""
        cards.append(
            f'<button type="button" class="setcard{selected_class}" '
            f'data-model="{safe_model}" onclick="window.__selectModel({model_json})">'
            f'{visual}<span>{escape(model_stem)}</span></button>'
        )
    return '<div class="setcard-grid">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

custom_css = """
#app_title h1 { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 20px; }
.title-super { color: #FFFFFF !important; }
.title-impersonator { color: #10B981 !important; }
.title-app { color: #EF4444 !important; }
.setcard-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr)); gap: 6px; }
.setcard { display: flex; flex-direction: column; align-items: center; gap: 5px; padding: 7px;
  color: #e2e8f0; background: #1e293b; border: 1px solid #475569; border-radius: 10px; cursor: pointer; }
.setcard:hover { border-color: #10b981; }
.setcard.selected { background: #064e3b; border-color: #34d399; }
.setcard-avatar { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; }
.setcard-initial { display: grid; place-items: center; background: #334155; font-size: 18px; font-weight: 700; }
.setcard span { overflow-wrap: anywhere; text-align: center; font-size: 11px; line-height: 1.2; }
.setcard-empty { padding: 12px; color: #94a3b8; }
body.dark, body, .gradio-container { background-color: #0f172a !important; color: #e2e8f0 !important; }
.gradio-container label, .gradio-container span, .gradio-container p { color: #cbd5e1 !important; }
input, textarea { background-color: #1e293b !important; color: #e2e8f0 !important; border-color: #475569 !important; }
"""

models_list = scan_rvc_models()
initial_model = models_list[0] if models_list else ""
try:
    input_devices, output_devices, default_input, default_output = get_audio_devices()
except Exception:
    input_devices, output_devices, default_input, default_output = [], [], None, None

force_dark_js = """() => {
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
  window.__selectModel = function(model) {
    const root = document.getElementById('hidden_model_trigger');
    if (!root) return;
    const input = root.querySelector('input, textarea');
    if (!input) return;
    input.value = model;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
}"""

with gr.Blocks(title="Super Impersonator App 3.0") as demo:
    gr.HTML(
        '<div id="app_title"><h1><span class="title-super">SUPER</span> '
        '<span class="title-impersonator">IMPERSONATOR</span> '
        '<span class="title-app">APP 3.0</span></h1></div>'
    )
    selected_model_state = gr.Textbox(
        value=initial_model, visible=False, elem_id="hidden_model_trigger", interactive=True
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Inferenz-Setup")
            input_audio = gr.Audio(
                label="Audioquelle",
                type="filepath",
                value=str(ENGINE_DIR / "audio.wav")
                if (ENGINE_DIR / "audio.wav").exists()
                else None,
                interactive=True,
            )
            gr.Markdown("#### RVC-Modell (.pth) wählen")
            setcards_container = gr.HTML(
                value=generate_setcards_html(models_list, initial_model)
            )
            setcard_convert_btn = gr.Button(
                "⚡ MIT GEWÄHLTER SETCARD UMwANDELN",
                variant="primary",
            )
            model_dropdown = gr.Dropdown(
                label="Modell (Alternativ-Auswahl)",
                choices=models_list,
                value=initial_model if initial_model else None,
            )

            with gr.Accordion("⚙️ RVC Kern-Einstellungen & Pitch-Autodetect", open=True):
                with gr.Row():
                    vocal_pitch = gr.Slider(
                        label="Pitch Shift (Halbtöne)", minimum=-24, maximum=24, step=1, value=0
                    )
                    auto_pitch_btn = gr.Button("🔍 PITCH AUTODETECT", variant="secondary")
                f0_method = gr.Dropdown(
                    label="F0 Methode", choices=["crepe", "rmvpe", "fcpe"], value="crepe"
                )
                index_rate = gr.Slider(label="Index Rate", minimum=0.0, maximum=1.0, value=1.0)
                protect = gr.Slider(label="Protect", minimum=0.0, maximum=0.5, value=0.33)
                filter_radius = gr.Slider(label="Glättungsradius", minimum=0, maximum=7, value=3)
                rms_mix_rate = gr.Slider(
                    label="Volume Envelope", minimum=0.0, maximum=1.0, value=0.25
                )
                hop_length = gr.Dropdown(
                    label="Hop Length (Crepe)", choices=["64", "128", "256"], value="128"
                )

            gr.Markdown("### 🎛️ Studio-Effektkette")
            studio_preset = gr.Dropdown(
                label="Effekt Presets",
                choices=list(STUDIO_PRESETS),
                value="Standard (Manuell)",
            )
            with gr.Accordion("🎙️ Eingangspegel & Dynamik (Gate/Comp)", open=False):
                vocal_gain = gr.Slider(
                    label="Mic Gain (dB)", minimum=-12, maximum=24, value=0.0
                )
                noise_red = gr.Slider(
                    label="Rauschunterdrückung (%)", minimum=0, maximum=100, value=0
                )
                gate_threshold = gr.Slider(
                    label="Gate Thresh (dB)", minimum=-60, maximum=0, value=-50
                )
                gate_release = gr.Slider(
                    label="Gate Release (ms)", minimum=50, maximum=1000, value=150
                )
                comp_thresh = gr.Slider(
                    label="Comp Thresh (dB)", minimum=-40, maximum=0, value=-20
                )
                comp_ratio = gr.Slider(
                    label="Comp Ratio", minimum=1.0, maximum=10.0, value=3.5
                )
                comp_attack = gr.Slider(
                    label="Comp Attack (ms)", minimum=1, maximum=100, value=15
                )
                comp_release = gr.Slider(
                    label="Comp Release (ms)", minimum=10, maximum=1000, value=200
                )
            with gr.Accordion("🎚️ Frequenz-Tuning (3-Band EQ)", open=False):
                eq_low = gr.Slider(label="Bass-Bereich (dB)", minimum=-12, maximum=12, value=0)
                eq_mid = gr.Slider(label="Stimm-Präsenz (dB)", minimum=-12, maximum=12, value=2)
                eq_high = gr.Slider(label="Brillanz (dB)", minimum=-12, maximum=12, value=1)
            with gr.Accordion("🌌 Raumklang & Zeiteffekte (Delay/Reverb)", open=False):
                echo_time = gr.Slider(label="Delay Time (s)", minimum=0.0, maximum=2.0, value=0.3)
                echo_feedback = gr.Slider(label="Delay Feedback (%)", minimum=0, maximum=100, value=30)
                reverb_size = gr.Slider(label="Reverb Room Size", minimum=0.0, maximum=1.0, value=0.5)
                fx_mix = gr.Slider(
                    label="Gesamter FX-Mixanteil (Wet %)",
                    minimum=0,
                    maximum=100,
                    value=15,
                )

        with gr.Column(scale=1):
            gr.Markdown("### 2. Live-Audio & Aufzeichnung")
            input_device = gr.Dropdown(
                label="Mikrofon-Eingang", choices=input_devices, value=default_input
            )
            output_device = gr.Dropdown(
                label="Monitor-Ausgang", choices=output_devices, value=default_output
            )
            live_block_size = gr.Slider(
                label="Buffer-Blockgröße", minimum=1024, maximum=4096, step=1024, value=2048
            )
            live_vu_meter = gr.Slider(
                label="Eingangspegel (dB)", minimum=-60, maximum=0, value=-60, interactive=False
            )
            vu_status_txt = gr.Markdown("OFFLINE")
            gr.Markdown("#### Hintergrundspur & Sidechain")
            bgm_file = gr.File(label="Hintergrundmusik (Beat/Track)", type="filepath")
            bgm_volume = gr.Slider(
                label="Musik Lautstärke (dB)", minimum=-40, maximum=12, value=0
            )
            ducking_intensity = gr.Slider(
                label="Sidechain Ducking (Absenkung dB)", minimum=0, maximum=30, value=12
            )
            ducking_attack = gr.Slider(
                label="Ducking Attack (ms)", minimum=5, maximum=250, step=5, value=20
            )
            ducking_release = gr.Slider(
                label="Ducking Release (ms)", minimum=50, maximum=1500, step=25, value=400
            )
            with gr.Row():
                start_rec_btn = gr.Button("REC START", variant="primary")
                stop_rec_btn = gr.Button("REC STOP", variant="stop")
            live_preview_audio = gr.Audio(label="Live-Aufnahme-Playback", interactive=False)

    gr.Markdown("### 3. Master Export & Format")
    export_codec = gr.Radio(["WAV", "MP3", "FLAC", "WMA"], value="WAV", label="Exportformat")
    mp3_bitrate = gr.Slider(
        label="Bitrate (kbps)", minimum=128, maximum=320, step=32, value=320
    )
    sample_rate = gr.Radio(["44100 Hz", "48000 Hz"], value="48000 Hz", label="Sample-Rate")
    bounce_btn = gr.Button("BOUNCE SKRIPT & MASTER EXPORT", variant="primary")
    output_audio = gr.Audio(label="Master-Ausgabedatei", type="filepath")
    status_output = gr.Textbox(label="System-Log", lines=4, value="System bereit.")

    def trigger_auto_pitch(audio, model):
        pitch, message = calculate_auto_pitch(audio, model)
        return gr.update(value=pitch), message

    auto_pitch_btn.click(
        fn=trigger_auto_pitch,
        inputs=[input_audio, selected_model_state],
        outputs=[vocal_pitch, status_output],
    )

    def on_model_select(model):
        if not model:
            return generate_setcards_html(models_list, None), gr.update()
        return generate_setcards_html(models_list, model), gr.update(value=model)

    selected_model_state.change(
        fn=on_model_select,
        inputs=[selected_model_state],
        outputs=[setcards_container, model_dropdown],
    )
    model_dropdown.change(
        fn=lambda model: model if model else gr.update(),
        inputs=[model_dropdown],
        outputs=[selected_model_state],
    )
    studio_preset.change(
        fn=load_preset_values,
        inputs=[studio_preset],
        outputs=[
            vocal_gain,
            noise_red,
            gate_threshold,
            gate_release,
            comp_thresh,
            comp_ratio,
            comp_attack,
            comp_release,
            eq_low,
            eq_mid,
            eq_high,
            echo_time,
            echo_feedback,
            reverb_size,
            fx_mix,
        ],
    )

    live_vu_timer = gr.Timer(value=0.1, active=True)
    live_vu_timer.tick(fn=update_live_vu, outputs=[live_vu_meter, vu_status_txt])
    start_rec_btn.click(
        fn=start_live_recording,
        inputs=[input_device, output_device, live_block_size],
        outputs=status_output,
    )
    stop_rec_btn.click(
        fn=stop_live_recording,
        inputs=[export_codec, mp3_bitrate],
        outputs=[live_preview_audio, status_output],
    )

    conversion_inputs = [
        input_audio,
        selected_model_state,
        vocal_pitch,
        f0_method,
        index_rate,
        protect,
        filter_radius,
        rms_mix_rate,
        hop_length,
        vocal_gain,
        noise_red,
        gate_threshold,
        gate_release,
        comp_thresh,
        comp_ratio,
        comp_attack,
        comp_release,
        eq_low,
        eq_mid,
        eq_high,
        echo_time,
        echo_feedback,
        reverb_size,
        fx_mix,
        export_codec,
        sample_rate,
        mp3_bitrate,
        bgm_file,
        bgm_volume,
        ducking_intensity,
        ducking_attack,
        ducking_release,
    ]
    bounce_btn.click(
        fn=process_audio,
        inputs=conversion_inputs,
        outputs=[output_audio, status_output],
        show_progress="full",
    )
    setcard_convert_btn.click(
        fn=process_audio,
        inputs=conversion_inputs,
        outputs=[output_audio, status_output],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="red",
            neutral_hue="slate",
        ),
        css=custom_css,
        js=force_dark_js,
    )