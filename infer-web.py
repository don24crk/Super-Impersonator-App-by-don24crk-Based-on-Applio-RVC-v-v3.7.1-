import contextlib
import importlib
import os
import re
import shutil
import sys
import threading
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf


# ---------------------------------------------------------------------------
# NumPy 2.x compatibility shim for Applio/RVC (and other legacy dependencies)
# ---------------------------------------------------------------------------
def _patch_numpy_compat():
    """Restore NumPy aliases removed in 1.24 / 2.0 for legacy code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

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

        try:
            import numpy.exceptions as np_exceptions

            if not hasattr(np_exceptions, "RankWarning"):
                rank_warning = type(
                    "RankWarning",
                    (Warning,),
                    {
                        "__module__": "numpy.exceptions",
                        "__doc__": "Polyfit may be poorly conditioned.",
                    },
                )
                np_exceptions.RankWarning = rank_warning
                setattr(np, "RankWarning", rank_warning)
        except Exception:
            pass


_patch_numpy_compat()

warnings.filterwarnings(
    "ignore",
    message=".*NumPy version.*required for this version of SciPy.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Gradio / Matplotlib backend crash guard
# ---------------------------------------------------------------------------
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
# Avatar-Suchpfade (erster Treffer gewinnt). Mitgelieferte Avatare liegen
# direkt im Projekt unter assets/avatars.
AVATAR_DIRS = [
    ENGINE_DIR / "assets" / "avatars",
    ENGINE_DIR / "avatare",
    AVATAR_DIR,
]

DEFAULT_ENV_DIR = Path(r"C:\Users\User\Downloads\ApplioV3.7.1\env")

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
                env_dir
                / "Lib"
                / "site-packages"
                / "torch"
                / "lib"
                / "torch_rt"
                / "dlls",
                env_dir
                / "Lib"
                / "site-packages"
                / "torch"
                / "lib"
                / "torch_rt"
                / "nvidia",
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


@contextlib.contextmanager
def _applio_working_dir():
    """Applio loads predictor weights (rmvpe.pt, fcpe.pt) relative to the CWD."""
    previous = os.getcwd()
    try:
        os.chdir(APPLIO_DIR)
        yield
    finally:
        os.chdir(previous)


def describe_applio_environment():
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
    preset = STUDIO_PRESETS.get(preset_name)
    if preset is None:
        return [gr.update() for _ in PRESET_OUTPUT_KEYS]
    return [gr.update(value=preset[key]) for key in PRESET_OUTPUT_KEYS]


# ---------------------------------------------------------------------------
# Audio and model helpers
# ---------------------------------------------------------------------------


def resolve_model_path(relative_path):
    """Resolve a relative model path to an absolute path."""
    if not relative_path:
        return None
    rel_path = Path(relative_path)
    if rel_path.is_absolute() and rel_path.is_file():
        return rel_path
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


def get_audio_devices():
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


_MODEL_META_CACHE = {}


def find_associated_index(model_rel_path):
    if not model_rel_path:
        return ""
    model_base = Path(model_rel_path).stem.lower()
    matches = []
    for directory in (MODELS_DIR, APPLIO_MODELS_DIR):
        if not directory.exists():
            continue
        for index_path in directory.rglob("*.index"):
            stem = index_path.stem.lower()
            if stem == model_base:
                score = 3
            elif stem.startswith(model_base) or model_base.startswith(stem):
                score = 2
            elif model_base in stem or stem in model_base:
                score = 1
            else:
                continue
            try:
                mtime = index_path.stat().st_mtime
            except OSError:
                mtime = 0
            matches.append((score, mtime, str(index_path)))
    if not matches:
        return ""
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return matches[0][2]


_PITCH_PATTERNS = [
    # "pitch:-12", "pitch 5", "transpose=+3"
    r"(?:pitch|transpose|transpos|semitones?|semi|key)\s*[:=]?\s*([+-]?\d{1,2})\s*(?:st|ht|semi|semitones?)?\b",
    # "+5st", "-3 ht", "4semi"
    r"([+-]?\d{1,2})\s*(?:st|ht|semi|semitones?)\b",
    # Kurzform in Dateinamen: "model_p12", "obama_P-5", "voice_p+4"
    r"(?:^|[^a-z0-9])p([+-]?\d{1,2})(?![0-9])",
]


def _guess_embedded_pitch(text):
    """Pitch-Angabe aus Modell-/Index-Namen oder Metadaten lesen."""
    if not text:
        return None
    text = str(text).lower()
    for pattern in _PITCH_PATTERNS:
        for match in re.finditer(pattern, text):
            value = int(match.group(1))
            if -24 <= value <= 24:
                return value
    return None


def inspect_model_metadata(model_rel_path):
    if model_rel_path in _MODEL_META_CACHE:
        return _MODEL_META_CACHE[model_rel_path]

    meta = {
        "sr": None,
        "version": "",
        "epoch": None,
        "f0": None,
        "embedded_pitch": 0,
        "index_path": "",
        "raw_info": "",
    }
    path = resolve_model_path(model_rel_path)
    if path is None:
        return meta

    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            cfg = ckpt.get("config")
            if isinstance(cfg, (list, tuple)) and len(cfg) > 0:
                try:
                    meta["sr"] = int(cfg[0])
                except (TypeError, ValueError):
                    pass
            if meta["sr"] is None and ckpt.get("sr") is not None:
                try:
                    meta["sr"] = int(ckpt["sr"])
                except (TypeError, ValueError):
                    pass
            meta["version"] = str(ckpt.get("version", "") or "")
            meta["epoch"] = ckpt.get("epoch")
            meta["f0"] = ckpt.get("f0")
            meta["raw_info"] = str(
                ckpt.get("info", "") or ckpt.get("name", "") or path.stem
            )

            if meta["sr"] is None:
                sr_match = re.search(
                    r"sample\s*rate\s*[:=]?\s*(\d{4,6})",
                    meta["raw_info"].lower(),
                )
                if sr_match:
                    meta["sr"] = int(sr_match.group(1))
    except Exception as exc:
        meta["raw_info"] = path.stem
        print(f"Metadaten konnten nicht gelesen werden ({path.name}): {exc}")

    # Standard-Pitch bestimmen: erst Modell-Metadaten/Name, dann Index-Dateiname.
    pitch = _guess_embedded_pitch(meta["raw_info"] + " " + path.stem)
    index_path = find_associated_index(model_rel_path)
    if pitch is None and index_path:
        pitch = _guess_embedded_pitch(Path(index_path).stem)
    meta["embedded_pitch"] = int(pitch) if pitch is not None else 0
    meta["index_path"] = index_path
    _MODEL_META_CACHE[model_rel_path] = meta
    return meta


def describe_model_selection(model_rel_path):
    if not model_rel_path:
        return "Kein Modell ausgewählt."
    meta = inspect_model_metadata(model_rel_path)
    index_path = find_associated_index(model_rel_path)
    sr_text = f"{meta['sr']} Hz" if meta["sr"] else "unbekannt"
    parts = [
        f"Modell geladen: SR={sr_text}",
        f"Standard-Pitch={meta['embedded_pitch']:+d} HT",
    ]
    if meta["version"]:
        parts.append(f"Version={meta['version']}")
    if meta["epoch"] is not None:
        parts.append(f"Epoch={meta['epoch']}")
    if meta["f0"] == 0:
        parts.append("⚠️ Modell ohne F0 trainiert – Pitch-Shift wirkungslos")
    parts.append("Index: " + (Path(index_path).name if index_path else "keiner gefunden"))
    avatar = find_avatar_file(Path(model_rel_path).stem)
    parts.append("Avatar: " + (avatar.name if avatar else "keiner"))
    return " | ".join(parts)


def make_output_path(directory, prefix, extension):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{timestamp}.{extension}"


def convert_codec(input_wav, target_codec, mp3_bitrate=320, output_dir=None):
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
    fx_enabled,
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
    if not fx_enabled:
        return input_wav_path

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
    rvc_index_path,
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
    fx_enabled,
    export_codec,
    sample_rate,
    mp3_bitrate,
    bgm_file,
    bgm_volume,
    ducking_intensity,
    ducking_attack,
    ducking_release,
    embedder_choice=None,
    speaker_id=0,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
    formant_shifting=False,
    formant_qfrency=1.0,
    formant_timbre=1.0,
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

    selected_index_path = Path(rvc_index_path) if rvc_index_path else None
    if selected_index_path is None or not selected_index_path.is_file():
        associated_index = find_associated_index(rvc_model_path)
        selected_index_path = Path(associated_index) if associated_index else None
    index_path_for_inference = (
        str(selected_index_path)
        if selected_index_path is not None and selected_index_path.is_file()
        else ""
    )

    try:
        target_sr = int("".join(filter(str.isdigit, str(sample_rate))))
        if target_sr <= 0:
            raise ValueError(f"Ungültige Sample-Rate: {sample_rate}")

        VC = load_vc_class()

        # ContentVec / HuBERT-Embedder bestimmen (Standard: ContentVec).
        embedder_model, embedder_custom = resolve_embedder(embedder_choice)
        embedder_info = embedder_model or "contentvec (Standard)"
        if embedder_custom:
            embedder_info = f"custom: {Path(embedder_custom).name}"
        # None-Werte NICHT uebergeben -> Applio-Default "contentvec" greift.
        embedder_kwargs = build_embedder_kwargs(embedder_model, embedder_custom)
        print(f"[INFO] Embedder fuer Inferenz: {embedder_info} | Kwargs: {embedder_kwargs}")

        # Sprecher-ID (Multi-Speaker-Modelle; 0 = Standard-Sprecher).
        try:
            speaker_id = max(0, int(speaker_id or 0))
        except (TypeError, ValueError):
            speaker_id = 0

        # Advanced Pitch & Tuning / Formantverschiebung.
        tuning_kwargs = {
            "proposed_pitch": bool(proposed_pitch),
            "proposed_pitch_threshold": float(proposed_pitch_threshold),
        }
        if formant_shifting:
            tuning_kwargs.update(
                formant_shifting=True,
                formant_qfrency=float(formant_qfrency),
                formant_timbre=float(formant_timbre),
            )
        print(
            f"[INFO] Sprecher-ID: {speaker_id} | Proposed Pitch: "
            f"{bool(proposed_pitch)} ({float(proposed_pitch_threshold)} Hz) | "
            f"Formant: {bool(formant_shifting)}"
        )

        vc = VC()
        raw_output_path = make_output_path(TEMP_DIR, "rvc_raw", "wav")
        with _applio_working_dir():
            if hasattr(vc, "convert_audio"):
                convert_kwargs = dict(
                    audio_input_path=str(input_path),
                    audio_output_path=str(raw_output_path),
                    model_path=str(model_path),
                    index_path=index_path_for_inference,
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
                    sid=speaker_id,
                )
                convert_kwargs.update(embedder_kwargs)
                convert_kwargs.update(tuning_kwargs)
                _call_with_supported_kwargs(vc.convert_audio, **convert_kwargs)
                rvc_wav_path = str(raw_output_path)
            else:
                # Legacy-API: get_vc + vc_single. Signatur-Reihenfolge von
                # get_vc unterscheidet sich je nach RVC-Version, daher beide
                # Aufrufvarianten absichern.
                try:
                    _call_with_supported_kwargs(
                        vc.get_vc,
                        weight_root=str(model_path),
                        sid=speaker_id,
                        **embedder_kwargs,
                    )
                except TypeError:
                    _call_with_supported_kwargs(
                        vc.get_vc, speaker_id, str(model_path), **embedder_kwargs
                    )
                vc_kwargs = dict(
                    sid=speaker_id,
                    input_audio_path=str(input_path),
                    f0_up_key=int(vocal_pitch),
                    f0_file=None,
                    f0_method=str(f0_method),
                    file_index=index_path_for_inference or None,
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
                vc_kwargs.update(embedder_kwargs)
                result = _call_with_supported_kwargs(vc.vc_single, **vc_kwargs)
                rvc_wav_path = normalize_rvc_result(result, target_sr)
        master_path = apply_studio_fx(
            rvc_wav_path,
            fx_enabled,
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
        return (
            final_path,
            "Studio-Inferenz erfolgreich abgeschlossen! "
            f"Modell: {Path(model_path).name} | Pitch: {int(vocal_pitch):+d} HT | "
            f"F0: {f0_method} | Index-Rate: {float(index_rate):.2f} | "
            f"Volume-Envelope: {float(rms_mix_rate):.2f} | "
            f"Protect: {float(protect):.2f} | Speaker-ID: {speaker_id} | "
            f"Proposed Pitch: {'an' if proposed_pitch else 'aus'} | "
            f"Formant: {'an' if formant_shifting else 'aus'} | "
            f"Embedder: {embedder_info} | SR: {target_sr} Hz",
        )
    except Exception as exc:
        tb_text = traceback.format_exc()
        print("=" * 70)
        print("[FEHLER] Inferenz-Pipeline")
        print(tb_text)
        print("=" * 70)
        detail = str(exc).strip()
        if not detail or detail == "None":
            detail = (
                "keine Fehlermeldung (haeufig KeyError(None)/TypeError(None) "
                "durch einen ungueltigen None-Parameter)"
            )
        last_frame = ""
        tb_lines = [line for line in tb_text.strip().splitlines() if line.strip()]
        if tb_lines:
            last_frame = tb_lines[-1].strip()
        message = f"Fehler in der Inferenz-Pipeline: {type(exc).__name__}: {detail}"
        if last_frame and last_frame not in message:
            message += f"\nLetzter Schritt: {last_frame}"
        message += "\nVollstaendiger Traceback wurde in der Konsole ausgegeben."
        return None, message


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


AVATAR_CACHE_DIR = ENGINE_DIR / ".avatar_cache"
AVATAR_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Bekannte Kuenstlernamen -> Avatar-Datei (Stem). Ergaenzt die automatische
# Zuordnung, wenn der Modellname vom Dateinamen abweicht.
AVATAR_ALIASES = {
    "obama": "obama",
    "barack": "obama",
    "barrack": "obama",
    "barackobama": "obama",
    "barrackobama": "obama",
    "timberlake": "justintimberlake",
    "justin": "justintimberlake",
    "justintimberlake": "justintimberlake",
    "jt": "justintimberlake",
    "michaeljackson": "mj",
    "michael": "mj",
    "jackson": "mj",
    "mj": "mj",
    "mj2": "mj2",
    "mjt": "mjthriller",
    "mjthriller": "mjthriller",
    "thriller": "mjthriller",
    "weeknd": "the_weeknd",
    "theweeknd": "the_weeknd",
    "abel": "the_weeknd",
    "connor": "connormaynard",
    "maynard": "connormaynard",
    "connormaynard": "connormaynard",
    "conor": "connormaynard",
    "cm": "connormaynard",
}

_NOISE_TOKENS = (
    "pth", "index", "model", "rvc", "v1", "v2", "v3", "final", "new", "best",
    "e100", "e200", "e300", "e400", "e500", "40k", "48k", "32k", "44k",
)


def _normalize_name(value):
    """Kleinschreibung, nur Buchstaben/Ziffern (fuer Namensvergleiche)."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _avatar_index():
    """Alle verfuegbaren Avatare: {normalisierter_stem: pfad}."""
    found = {}
    for directory in AVATAR_DIRS:
        try:
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.iterdir()):
                if file_path.suffix.lower() not in AVATAR_EXTENSIONS:
                    continue
                if not file_path.is_file():
                    continue
                key = _normalize_name(file_path.stem)
                if key and key not in found:
                    found[key] = file_path
        except OSError as exc:
            print(f"Avatar-Ordner nicht lesbar ({directory}): {exc}")
    return found


def find_avatar_file(model_stem):
    """Passenden Avatar zu einem Modellnamen finden (exakt, Alias, unscharf)."""
    if not model_stem:
        return None
    available = _avatar_index()
    if not available:
        return None

    stem_norm = _normalize_name(model_stem)
    if not stem_norm:
        return None

    # 1) exakter Treffer
    if stem_norm in available:
        return available[stem_norm]

    # 2) Alias-Treffer (Modellname enthaelt bekannten Kuenstlernamen).
    #    Laengste Aliasse zuerst, damit "mjthriller" nicht auf "mj" faellt.
    for alias, target in sorted(
        AVATAR_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias in stem_norm:
            target_norm = _normalize_name(target)
            if target_norm in available:
                return available[target_norm]

    # 3) unscharf: Modellname beginnt mit Avatar-Namen oder umgekehrt
    candidates = []
    for key, path in available.items():
        if len(key) < 3:
            continue
        if stem_norm.startswith(key) or key.startswith(stem_norm):
            candidates.append((len(key), path))
        elif key in stem_norm or stem_norm in key:
            candidates.append((len(key) - 2, path))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # 4) Wort-Treffer (z. B. "miley_cyrus_v2" -> "miley")
    model_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(model_stem).lower())
        if len(token) >= 3 and token not in _NOISE_TOKENS
    }
    for key, path in available.items():
        if key in model_tokens:
            return path
    return None


def get_avatar_icon_path(model_stem):
    """Avatar in einen lokalen Cache kopieren und einen KURZEN Pfad liefern.

    Gradios gr.Button(icon=...) erwartet einen Dateipfad. Eine Data-URI wird
    als Pfad interpretiert und laesst Windows mit "path too long" abstuerzen,
    daher wird hier eine echte (kurze) Datei im Cache-Verzeichnis verwendet.
    """
    source = find_avatar_file(model_stem)
    if source is None:
        return None
    try:
        AVATAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_stem))[:40] or "avatar"
        target = AVATAR_CACHE_DIR / f"{safe_stem}{source.suffix.lower()}"
        if (
            not target.is_file()
            or target.stat().st_mtime < source.stat().st_mtime
            or target.stat().st_size != source.stat().st_size
        ):
            shutil.copyfile(source, target)
        if target.is_file() and target.stat().st_size > 0:
            # Sicherheitsnetz gegen Windows "path too long".
            if len(str(target)) < 200:
                return str(target)
    except OSError as exc:
        print(f"Avatar konnte nicht geladen werden ({model_stem}): {exc}")
    return None


# ---------------------------------------------------------------------------
# ContentVec / HuBERT embedder support
# ---------------------------------------------------------------------------

EMBEDDER_DIRS = [
    APPLIO_DIR / "rvc" / "models" / "embedders",
    APPLIO_DIR / "assets" / "embedders",
    APPLIO_DIR / "assets" / "hubert",
    APPLIO_DIR / "rvc" / "models",
    ENGINE_DIR / "assets" / "embedders",
    ENGINE_DIR / "assets" / "hubert",
]

_EMBEDDER_NAME_HINTS = (
    "contentvec",
    "hubert",
    "wav2vec",
    "spin",
    "whisper",
    "cnhubert",
    "vec",
)

# Bekannte Embedder-Kurznamen, die Applio direkt kennt.
EMBEDDER_PRESETS = ["contentvec", "hubert_base", "spin", "whisper"]


def find_embedder_models():
    """Alle lokal vorhandenen ContentVec-/HuBERT-Modelle finden.

    Rueckgabe: Liste von (anzeige_name, absoluter_pfad).
    """
    found = {}
    for directory in EMBEDDER_DIRS:
        try:
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.rglob("*")):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in (".pt", ".pth", ".bin", ".safetensors"):
                    continue
                name_lower = file_path.stem.lower()
                if not any(hint in name_lower for hint in _EMBEDDER_NAME_HINTS):
                    continue
                key = name_lower
                if key not in found:
                    found[key] = file_path
        except OSError as exc:
            print(f"Embedder-Ordner nicht lesbar ({directory}): {exc}")
    return [(path.stem, str(path)) for _, path in found.items()]


def resolve_embedder(choice):
    """Auswahl in (embedder_model, embedder_model_custom) uebersetzen.

    - "auto"          -> (None, None): Applio nutzt seinen Standard (ContentVec)
    - bekannter Name  -> (name, None)
    - Dateipfad       -> ("custom", pfad)

    Hinweis: (None, None) bedeutet *nicht* "None uebergeben", sondern "Kwarg
    weglassen", damit Applio seinen eigenen Default (embedder_model="contentvec")
    verwendet. Siehe build_embedder_kwargs().
    """
    choice_text = str(choice or "").strip().lower()
    if not choice_text or choice_text.startswith(("auto", "standard", "default")):
        return None, None

    choice = str(choice).strip()
    path = Path(choice)
    if path.is_file():
        return "custom", str(path)

    lowered = choice.lower()
    for preset in EMBEDDER_PRESETS:
        if lowered == preset or lowered.startswith(preset):
            return preset, None

    # Anzeigename (Datei-Stem) -> echten Pfad suchen
    for name, file_path in find_embedder_models():
        if name.lower() == lowered:
            return "custom", file_path

    return None, None


def build_embedder_kwargs(embedder_model, embedder_custom):
    """Nur tatsaechlich gesetzte Embedder-Kwargs zurueckgeben.

    Applio definiert `embedder_model: str = "contentvec"`. Wird hier explizit
    `embedder_model=None` uebergeben, ueberschreibt das den Default und die
    interne `embedding_list[None]`-Suche wirft `KeyError(None)` -> die
    Fehlermeldung erscheint nur als "None". Deshalb: None-Werte weglassen und
    Applio seinen Default benutzen lassen.
    """
    kwargs = {}
    if embedder_model:
        kwargs["embedder_model"] = str(embedder_model)
    if embedder_custom:
        kwargs["embedder_model_custom"] = str(embedder_custom)
    return kwargs


def _call_with_supported_kwargs(func, **kwargs):
    """Nur Kwargs uebergeben, die die Ziel-Funktion wirklich akzeptiert.

    Wichtig: Nur die Signatur-Auswertung wird gegen Fehler abgesichert. Ein
    TypeError, der *innerhalb* der Ziel-Funktion entsteht, wird NICHT mehr
    verschluckt und mit allen Kwargs erneut aufgerufen (frueher maskierte das
    die eigentliche Ursache und produzierte kryptische Folgefehler).
    """
    try:
        import inspect

        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # Builtins / C-Funktionen ohne introspektierbare Signatur.
        return func(**kwargs)

    accepts_var_kw = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kw:
        return func(**kwargs)
    allowed = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    dropped = sorted(set(kwargs) - set(allowed))
    if dropped:
        print(f"[INFO] Nicht unterstuetzte Parameter ignoriert: {', '.join(dropped)}")
    return func(**allowed)


def _make_setcard_button(label, icon_path):
    """Modell-Button bauen, der mit jeder Gradio-Version funktioniert.

    Neuere Versionen unterstuetzen icon/size/elem_classes; fehlt ein Kwarg,
    wird schrittweise darauf verzichtet, statt den App-Start abzubrechen.
    """
    attempts = [
        {"value": label, "icon": icon_path, "elem_classes": "setcard-btn", "size": "sm"},
        {"value": label, "icon": icon_path, "elem_classes": "setcard-btn"},
        {"value": label, "elem_classes": "setcard-btn"},
        {"value": label},
    ]
    last_exc = None
    for kwargs in attempts:
        if kwargs.get("icon") is None:
            kwargs.pop("icon", None)
        try:
            return gr.Button(**kwargs)
        except Exception as exc:  # TypeError bei unbekannten Kwargs
            last_exc = exc
    raise last_exc


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

custom_css = """
#app_title h1 { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 20px; }
.title-super { color: #FFFFFF !important; }
.title-impersonator { color: #10B981 !important; }
.title-app { color: #EF4444 !important; }
/* Modell-Auswahl: native Gradio-Buttons als Karten-Grid */
.setcard-grid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)) !important;
  gap: 6px !important; flex-wrap: wrap !important; }
.setcard-btn { min-width: 0 !important; width: 100% !important; height: auto !important;
  display: flex !important; flex-direction: column !important; align-items: center !important;
  justify-content: center !important; gap: 4px !important; padding: 7px 5px !important;
  font-size: 11px !important; line-height: 1.2 !important; white-space: normal !important;
  border-radius: 10px !important; overflow-wrap: anywhere !important; text-align: center !important; }
.setcard-btn img { width: 44px !important; height: 44px !important; border-radius: 50% !important;
  object-fit: cover !important; margin: 0 auto 4px auto !important; display: block !important; }
.model-avatar { max-width: 150px !important; margin: 4px auto 8px auto !important; }
.model-avatar img { border-radius: 12px !important; object-fit: cover !important; }
.setcard-empty { padding: 12px; color: #94a3b8; }
body.dark, body, .gradio-container { background-color: #0f172a !important; color: #e2e8f0 !important; }
.gradio-container label, .gradio-container span, .gradio-container p { color: #cbd5e1 !important; }
input, textarea { background-color: #1e293b !important; color: #e2e8f0 !important; border-color: #475569 !important; }
"""

# Zusaetzliches CSS fuer die versteckten State-Textboxen
custom_css += """
#hidden_model_trigger, #hidden_index_trigger { display: none !important; }
#model_dropdown { margin-bottom: 6px; }
"""


models_list = scan_rvc_models()
initial_model = models_list[0] if models_list else ""
initial_index = find_associated_index(initial_model) if initial_model else ""
initial_meta = inspect_model_metadata(initial_model) if initial_model else {}
initial_pitch_value = int(initial_meta.get("embedded_pitch") or 0)  # Standard-Pitch des Startmodells
initial_status_value = (
    describe_model_selection(initial_model) if initial_model else "Kein Modell ausgewählt."
)
try:
    input_devices, output_devices, default_input, default_output = get_audio_devices()
except Exception:
    input_devices, output_devices, default_input, default_output = [], [], None, None

force_dark_js = """() => {
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
}"""

with gr.Blocks(title="Super Impersonator App 3.0") as demo:
    gr.HTML(
        '<div id="app_title"><h1><span class="title-super">SUPER</span> '
        '<span class="title-impersonator">IMPERSONATOR</span> '
        '<span class="title-app">APP 3.0</span></h1></div>'
    )
    selected_model_state = gr.Textbox(
        value=initial_model,
        elem_id="hidden_model_trigger",
        interactive=True,
        label="",
        container=False,
    )
    selected_index_state = gr.Textbox(
        value=initial_index,
        elem_id="hidden_index_trigger",
        interactive=False,
        label="",
        container=False,
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
            model_dropdown = gr.Dropdown(
                label="RVC-Modell (.pth)",
                choices=models_list,
                value=initial_model or None,
                interactive=True,
                elem_id="model_dropdown",
            )
            _avatar_kwargs = {
                "label": "Modell-Avatar",
                "value": get_avatar_icon_path(Path(initial_model).stem) if initial_model else None,
                "height": 150,
                "interactive": False,
                "elem_classes": "model-avatar",
            }
            try:
                model_avatar_preview = gr.Image(**_avatar_kwargs)
            except Exception as _avatar_exc:
                # Aeltere Gradio-Versionen kennen einzelne Kwargs nicht.
                print(f"Avatar-Vorschau mit reduzierten Optionen: {_avatar_exc}")
                model_avatar_preview = gr.Image(
                    label="Modell-Avatar",
                    value=_avatar_kwargs["value"],
                    interactive=False,
                )
            model_buttons = []
            if models_list:
                with gr.Row(elem_classes="setcard-grid"):
                    for _model in models_list:
                        _stem = Path(_model).stem
                        _btn = _make_setcard_button(
                            _stem, get_avatar_icon_path(_stem)
                        )
                        model_buttons.append((_model, _btn))
            else:
                gr.Markdown(
                    '<div class="setcard-empty">Keine .pth-Modelle im Verzeichnis gefunden.</div>'
                )
            setcard_convert_btn = gr.Button(
                "⚡ MIT GEWÄHLTER SETCARD UMWANDELN",
                variant="primary",
            )

            with gr.Accordion("⚙️ RVC Kern-Einstellungen", open=True):
                gr.Markdown("#### 🎤 Stimme & Tonhöhe")
                vocal_pitch = gr.Slider(
                    label="Pitch Shift (Halbtöne)",
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=initial_pitch_value,
                )
                f0_method = gr.Dropdown(
                    label="F0 Methode",
                    choices=["crepe", "rmvpe", "fcpe"],
                    value="rmvpe",
                )
                _embedder_choices = ["auto (ContentVec)"] + [
                    name for name, _ in find_embedder_models()
                ]
                try:
                    embedder_choice = gr.Dropdown(
                        label="ContentVec / Embedder",
                        choices=_embedder_choices,
                        value=_embedder_choices[0],
                        info=(
                            "auto = Applio-Standard (ContentVec). Weitere Modelle "
                            "werden aus den embedders-Ordnern geladen."
                        ),
                    )
                except Exception:
                    # Aeltere Gradio-Versionen kennen "info" nicht.
                    embedder_choice = gr.Dropdown(
                        label="ContentVec / Embedder (auto = ContentVec)",
                        choices=_embedder_choices,
                        value=_embedder_choices[0],
                    )
                gr.Markdown("#### 🔀 Stimmabgleich & Algorithmen")
                index_rate = gr.Slider(
                    label="Index Rate",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.75,
                )
                protect = gr.Slider(
                    label="Stimmlose Konsonanten schützen (Protect)",
                    minimum=0.0,
                    maximum=0.5,
                    value=0.5,
                )
                filter_radius = gr.Slider(
                    label="Glättungsradius",
                    minimum=0,
                    maximum=7,
                    value=3,
                )
                rms_mix_rate = gr.Slider(
                    label="Lautstärkehüllkurve (Volume Envelope)",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                )
                hop_length = gr.Dropdown(
                    label="Hop Length (Crepe)",
                    choices=["64", "128", "256"],
                    value="128",
                )
                speaker_id = gr.Slider(
                    label="Sprecher-ID (Speaker ID, 0 = Standard)",
                    minimum=0,
                    maximum=20,
                    step=1,
                    value=0,
                )

            with gr.Accordion("🎯 Advanced Pitch & Tuning", open=False):
                proposed_pitch = gr.Checkbox(
                    label="Vorgeschlagene Tonhöhe (Proposed Pitch)",
                    value=False,
                    info="Automatische Tonhöhen-Korrektur während der Inferenz.",
                )
                proposed_pitch_threshold = gr.Slider(
                    label="Proposed-Pitch-Schwelle (Hz)",
                    minimum=50.0,
                    maximum=400.0,
                    step=5.0,
                    value=155.0,
                )
                formant_shifting = gr.Checkbox(
                    label="Formantverschiebung (Formant Shifting)",
                    value=False,
                    info="Verschiebt die Formanten unabhängig von der Tonhöhe.",
                )
                formant_qfrency = gr.Slider(
                    label="Formant Q-Frequenz",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                )
                formant_timbre = gr.Slider(
                    label="Formant Timbre",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                )

            gr.Markdown("### 🎛️ Studio-Effektkette")
            fx_enabled = gr.Checkbox(
                label="Effektkette einschalten",
                value=True,
                info="Ausschalten, um das RVC-Ergebnis ohne Studio-Effekte zu exportieren.",
            )
            studio_preset = gr.Dropdown(
                label="Effekt Presets",
                choices=list(STUDIO_PRESETS),
                value="Standard (Manuell)",
            )
            with gr.Accordion("🎙️ Eingangspegel & Dynamik (Gate/Comp)", open=False):
                vocal_gain = gr.Slider(
                    label="Mic Gain (dB)",
                    minimum=-12,
                    maximum=24,
                    value=0.0,
                )
                noise_red = gr.Slider(
                    label="Rauschunterdrückung (%)",
                    minimum=0,
                    maximum=100,
                    value=0,
                )
                gate_threshold = gr.Slider(
                    label="Gate Thresh (dB)",
                    minimum=-60,
                    maximum=0,
                    value=-50,
                )
                gate_release = gr.Slider(
                    label="Gate Release (ms)",
                    minimum=50,
                    maximum=1000,
                    value=150,
                )
                comp_thresh = gr.Slider(
                    label="Comp Thresh (dB)",
                    minimum=-40,
                    maximum=0,
                    value=-20,
                )
                comp_ratio = gr.Slider(
                    label="Comp Ratio",
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                )
                comp_attack = gr.Slider(
                    label="Comp Attack (ms)",
                    minimum=1,
                    maximum=100,
                    value=15,
                )
                comp_release = gr.Slider(
                    label="Comp Release (ms)",
                    minimum=1,
                    maximum=1000,
                    value=200,
                )
            with gr.Accordion("🎚️ Frequenz-Tuning (3-Band EQ)", open=False):
                eq_low = gr.Slider(
                    label="Bass-Bereich (dB)",
                    minimum=-12,
                    maximum=12,
                    value=0,
                )
                eq_mid = gr.Slider(
                    label="Stimm-Präsenz (dB)",
                    minimum=-12,
                    maximum=12,
                    value=2,
                )
                eq_high = gr.Slider(
                    label="Brillanz (dB)",
                    minimum=-12,
                    maximum=12,
                    value=1,
                )
            with gr.Accordion("🌌 Raumklang & Zeiteffekte (Delay/Reverb)", open=False):
                echo_time = gr.Slider(
                    label="Delay Time (s)",
                    minimum=0.0,
                    maximum=2.0,
                    value=0.3,
                )
                echo_feedback = gr.Slider(
                    label="Delay Feedback (%)",
                    minimum=0,
                    maximum=100,
                    value=30,
                )
                reverb_size = gr.Slider(
                    label="Reverb Room Size",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                )
                fx_mix = gr.Slider(
                    label="Gesamter FX-Mixanteil (Wet %)",
                    minimum=0,
                    maximum=100,
                    value=15,
                )

        with gr.Column(scale=1):
            gr.Markdown("### 2. Live-Audio & Aufzeichnung")
            input_device = gr.Dropdown(
                label="Mikrofon-Eingang",
                choices=input_devices,
                value=default_input,
            )
            output_device = gr.Dropdown(
                label="Monitor-Ausgang",
                choices=output_devices,
                value=default_output,
            )
            live_block_size = gr.Slider(
                label="Buffer-Blockgröße",
                minimum=1024,
                maximum=4096,
                step=1024,
                value=2048,
            )
            live_vu_meter = gr.Slider(
                label="Eingangspegel (dB)",
                minimum=-60,
                maximum=0,
                value=-60,
                interactive=False,
            )
            vu_status_txt = gr.Markdown("OFFLINE")
            gr.Markdown("#### Hintergrundspur & Sidechain")
            bgm_file = gr.File(label="Hintergrundmusik (Beat/Track)", type="filepath")
            bgm_volume = gr.Slider(
                label="Musik Lautstärke (dB)",
                minimum=-40,
                maximum=12,
                value=0,
            )
            ducking_intensity = gr.Slider(
                label="Sidechain Ducking (Absenkung dB)",
                minimum=0,
                maximum=30,
                value=12,
            )
            ducking_attack = gr.Slider(
                label="Ducking Attack (ms)",
                minimum=5,
                maximum=250,
                step=5,
                value=20,
            )
            ducking_release = gr.Slider(
                label="Ducking Release (ms)",
                minimum=50,
                maximum=1500,
                step=25,
                value=400,
            )
            with gr.Row():
                start_rec_btn = gr.Button("REC START", variant="primary")
                stop_rec_btn = gr.Button("REC STOP", variant="stop")
            live_preview_audio = gr.Audio(
                label="Live-Aufnahme-Playback",
                interactive=False,
            )

    gr.Markdown("### 3. Master Export & Format")
    export_codec = gr.Radio(
        ["WAV", "MP3", "FLAC", "WMA"],
        value="WAV",
        label="Exportformat",
    )
    mp3_bitrate = gr.Slider(
        label="Bitrate (kbps)",
        minimum=128,
        maximum=320,
        step=32,
        value=320,
    )
    sample_rate = gr.Radio(
        ["40000 Hz", "44100 Hz", "48000 Hz"],
        value="48000 Hz",
        label="Sample-Rate",
    )
    bounce_btn = gr.Button("BOUNCE SKRIPT & MASTER EXPORT", variant="primary")
    output_audio = gr.Audio(label="Master-Ausgabedatei", type="filepath")
    status_output = gr.Textbox(label="System-Log", lines=4, value=initial_status_value)

    def on_model_select(model):
        """Modell uebernehmen: Standard-Pitch, Index und Sample-Rate des Modells.

        Es gelten die im Modell hinterlegten Standardwerte:
        - Pitch  = Standard-Pitch aus .pth-Metadaten bzw. Index-Dateiname
        - Index  = automatisch zugeordnete .index-Datei des Modells
        - SR     = im Modell trainierte Sample-Rate (sonst 48000 Hz)
        """
        if not model:
            return (
                gr.update(),
                "",
                gr.update(),
                gr.update(),
                "Kein Modell ausgewählt.",
                gr.update(),
                gr.update(value=None, visible=False),
            )
        meta = inspect_model_metadata(model)
        index_path = find_associated_index(model)
        pitch_value = int(meta.get("embedded_pitch") or 0)
        model_sr = int(meta.get("sr") or 0)
        supported_rates = (40000, 44100, 48000)
        if model_sr > 0:
            nearest_sr = min(supported_rates, key=lambda rate: abs(rate - model_sr))
        else:
            nearest_sr = 48000
        sr_value = f"{nearest_sr} Hz"
        avatar_path = get_avatar_icon_path(Path(model).stem)
        return (
            gr.update(value=model),
            index_path,
            gr.update(value=pitch_value),
            gr.update(value=sr_value),
            describe_model_selection(model),
            gr.update(value=model),
            gr.update(value=avatar_path, visible=bool(avatar_path)),
        )

    model_select_outputs = [
        selected_model_state,
        selected_index_state,
        vocal_pitch,
        sample_rate,
        status_output,
        model_dropdown,
        model_avatar_preview,
    ]

    # Native Gradio-Events: jeder Karten-Button kennt sein Modell fest.
    def _bind_model_button(model_path):
        def _handler():
            return on_model_select(model_path)

        return _handler

    for _model_path, _model_button in model_buttons:
        _model_button.click(
            fn=_bind_model_button(_model_path),
            inputs=[],
            outputs=model_select_outputs,
        )

    model_dropdown.change(
        fn=on_model_select,
        inputs=[model_dropdown],
        outputs=model_select_outputs,
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
        selected_index_state,
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
        fx_enabled,
        export_codec,
        sample_rate,
        mp3_bitrate,
        bgm_file,
        bgm_volume,
        ducking_intensity,
        ducking_attack,
        ducking_release,
        embedder_choice,
        speaker_id,
        proposed_pitch,
        proposed_pitch_threshold,
        formant_shifting,
        formant_qfrency,
        formant_timbre,
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


def _find_free_port(start_port: int = 6969, attempts: int = 50) -> int:
    """Return the first free TCP port at or after start_port."""
    import socket

    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            env_port = int(env_port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", env_port))
            return env_port
        except (ValueError, OSError):
            print(
                f"[WARN] GRADIO_SERVER_PORT={env_port} is invalid or busy; "
                f"searching for a free port instead."
            )

    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"No free port found in range {start_port}-{start_port + attempts - 1}. "
        f"Close the process using these ports or set GRADIO_SERVER_PORT."
    )


if __name__ == "__main__":
    _port = _find_free_port(6969)
    print(f"[INFO] Starting Gradio server on http://127.0.0.1:{_port}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=_port,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="red",
            neutral_hue="slate",
        ),
        css=custom_css,
        js=force_dark_js,
    )
