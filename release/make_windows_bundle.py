"""release/make_windows_bundle.py - the Windows release folder, dist/strata-nvfp4 (zip it afterwards).

    python release/make_windows_bundle.py          (after building build-release: STRATA_PORTABLE=ON, sm_120a)

Takes the portable engine from build-release/, cuBLAS from %CUDA_PATH%, the part of llama.cpp the converter
imports from third_party/llama.cpp, and the bundle's own files (README, scripts, config) from release/windows/.
"""
import json
import os
import pathlib
import shutil
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "dist" / "strata-nvfp4"
CUDA = pathlib.Path(os.environ["CUDA_PATH"])
CUDA_BIN = CUDA / "bin" / "x64" if (CUDA / "bin" / "x64" / "cublas64_13.dll").exists() else CUDA / "bin"
LLAMA = REPO / "third_party" / "llama.cpp"
VERSION = "0.1.24-nvfp4.2"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
skip = shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "*_fake_server.py", "chat_golden.json", "tests")


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True).stdout.strip()


# engine: the portable build and the cuBLAS it links (the CUDA runtime is static)
cp(REPO / "build-release" / "strata.exe", OUT / "engine" / "strata.exe")
for dll in ("cublas64_13.dll", "cublasLt64_13.dll"):
    cp(CUDA_BIN / dll, OUT / "engine" / dll)
(OUT / "engine" / "BUILD.json").write_text(json.dumps({
    "version": VERSION, "source": "release", "archs": [120], "ptx": False, "cuda": CUDA.name.lstrip("v"),
    "vision": "none", "portable": True, "fork": "https://github.com/sergqwer/strata-nvfp4",
    "commit": git("rev-parse", "--short", "HEAD")}, indent=1) + "\n")

# the server (with its chat page) and the tools
shutil.copytree(REPO / "serve", OUT / "serve", ignore=skip)
shutil.copytree(REPO / "tools", OUT / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "vision"))

# the part of llama.cpp the converter imports, at its pinned commit, with its license
for name in ("convert_hf_to_gguf.py", "LICENSE"):
    cp(LLAMA / name, OUT / "third_party" / "llama.cpp" / name)
shutil.copytree(LLAMA / "conversion", OUT / "third_party" / "llama.cpp" / "conversion", ignore=skip)
shutil.copytree(LLAMA / "gguf-py", OUT / "third_party" / "llama.cpp" / "gguf-py",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples"))
(OUT / "third_party" / "llama.cpp" / "COMMIT").write_text(git("rev-parse", "HEAD", cwd=LLAMA) + "\n")

# data, docs, license
for name in ("expert-profile.bin", "draft_vocab.bin"):
    cp(REPO / "data" / name, OUT / "data" / name)
cp(REPO / "docs" / "NVFP4.md", OUT / "docs" / "NVFP4.md")
cp(REPO / "LICENSE", OUT / "LICENSE")

# the bundle's own files; .cmd with CRLF, or cmd.exe mangles them
for src in (REPO / "release" / "windows").rglob("*"):
    if src.is_file():
        dst = OUT / src.relative_to(REPO / "release" / "windows")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".cmd":
            dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        else:
            shutil.copy2(src, dst)

files = [f for f in OUT.rglob("*") if f.is_file()]
print("assembled %s: %d files, %.1f MB" % (OUT, len(files), sum(f.stat().st_size for f in files) / 1e6))
