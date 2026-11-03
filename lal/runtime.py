"""Start a local model server (Ollama) on Colab or on your own machine.

Every notebook calls `setup()` once. On Colab it installs Ollama, starts it in the
background and downloads the models. On a laptop it uses the Ollama you already run.
"""
import os
import shutil
import subprocess
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MAIN_MODEL = "llama3.1:8b"     # Meta Llama 3.1 8B: tool calling + JSON schema, ~5 GB
SMALL_MODEL = "llama3.2:3b"    # Meta Llama 3.2 3B: fast, ~2 GB


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], capture_output=True).returncode == 0


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def install_ollama():
    if shutil.which("ollama"):
        return
    print("Installing Ollama (about 1 minute)...")
    subprocess.run("apt-get -qq update && apt-get -qq install -y zstd pciutils", shell=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1", shell=True, check=True)


def start_server():
    if server_up():
        return
    # Ollama's defaults are a 4096-token context and 1 request at a time. The OpenAI-style API can't change
    # the context per call, so we set it when the server starts. 8k x 2 fits a 16 GB T4 with an 8B model.
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_KEEP_ALIVE": "30m",
           "OLLAMA_CONTEXT_LENGTH": os.getenv("OLLAMA_CONTEXT_LENGTH", "8192"),
           "OLLAMA_NUM_PARALLEL": os.getenv("OLLAMA_NUM_PARALLEL", "2")}
    subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log", "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        if server_up():
            return
        time.sleep(1)
    raise RuntimeError("Ollama did not start. See /tmp/ollama.log")


def pull(model: str):
    print(f"Downloading {model} (first time only)...")
    subprocess.run(["ollama", "pull", model], check=True)      # shows download progress


def warm_up(model: str):
    """Load the model into GPU memory now, so the first lesson call isn't slow (the first load can take ~1-2 minutes)."""
    import json
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps({"model": model, "prompt": "hi",
                                 "stream": False, "options": {"num_predict": 1}}).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        print("warm-up skipped:", e)


def setup(models=None) -> str:
    """Get a local model ready. Returns the name of the main model to use."""
    t0 = time.time()
    gpu = has_gpu()
    if in_colab():
        install_ollama()
    elif not shutil.which("ollama") and not server_up():
        raise RuntimeError("Install Ollama first: https://ollama.com/download")
    start_server()   # if a server was already running, restart the runtime to apply the settings above
    main = os.getenv("LAL_MODEL") or (MAIN_MODEL if gpu or not in_colab() else SMALL_MODEL)
    if in_colab() and not gpu:
        print("WARNING: no GPU found. Using the small model; it will be slower.\n"
              "Fix: Runtime > Change runtime type > T4 GPU, then rerun this cell.")
    wanted = list(dict.fromkeys([main, *(models or [])]))
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    for m in wanted:
        if (m if ":" in m else m + ":latest") not in have:
            pull(m)
    os.environ["LAL_MODEL"] = main
    warm_up(main)
    print(f"Ready in {time.time() - t0:.0f}s. Main model: {main}. GPU: {'yes' if gpu else 'no'}. "
          f"Everything runs on this machine; no data leaves it.")
    return main
