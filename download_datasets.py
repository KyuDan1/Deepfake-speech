#!/usr/bin/env python3
"""
Deepfake Speech Detection - Dataset Downloader

Downloads evaluation datasets:
- ASVspoof 2021 DF (DeepFake track) -> via Kaggle API
- In-The-Wild dataset
- WaveFake dataset (Generated + Reference: LJSpeech, JSUT)

Usage:
    python download_datasets.py --dataset all
    python download_datasets.py --dataset asvspoof2021
    python download_datasets.py --dataset inthewild
    python download_datasets.py --dataset wavefake

    # Parallel download (for 'all')
    python download_datasets.py --dataset all --parallel
"""

import argparse
import os
import subprocess
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Queue
import time


# Default paths
DEFAULT_DATA_DIR = Path("/mnt/tmp/Deepfake-speech/data")


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_wget(url: str, output_path: Path):
    """Download file using wget (better for large files)"""
    cmd = ["wget", "-c", "--progress=bar:force", "-O", str(output_path), url]
    
    # owncloud나 특정 사이트 인증서 문제 무시 옵션 추가
    if "owncloud" in url:
        cmd.append("--no-check-certificate")
        
    subprocess.run(cmd, check=True)

def download_asvspoof2021_df(data_dir: Path):
    """
    Download ASVspoof 2021 DF (DeepFake) evaluation dataset via Kaggle API
    Target: https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021
    """
    print("=" * 70)
    print("[ASVspoof2021] Starting download via Kaggle API...")
    print("=" * 70)

    target_dir = data_dir / "ASVspoof2021"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle Dataset Identifier
    kaggle_dataset = "mohammedabdeldayem/avsspoof-2021"

    # 1. Download via Kaggle API (With Retry Logic)
    print(f"[ASVspoof2021] [1/3] Downloading & Unzipping from Kaggle...")
    
    # Check if 'flac' directory already exists
    if (target_dir / "flac").exists() or (target_dir / "ASVspoof2021_DF_eval").exists():
         print(f"[ASVspoof2021]   [SKIP] Data seems to exist already.")
    else:
        # --- RETRY LOGIC START ---
        max_retries = 10
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[ASVspoof2021]   Attempt {attempt}/{max_retries}...")

                # kaggle datasets download -d [slug] --unzip
                # --path 옵션 없이 현재 디렉토리에 다운로드 후 이동
                cmd = [
                    "kaggle", "datasets", "download",
                    "-d", kaggle_dataset,
                    "--unzip"
                ]
                subprocess.run(cmd, check=True, cwd=str(target_dir))
                success = True
                print("[ASVspoof2021]   [OK] Download & Unzip complete")
                break # 성공하면 반복문 탈출

            except FileNotFoundError:
                print("[ASVspoof2021]   [ERROR] 'kaggle' command not found. Please install it (pip install kaggle).")
                return False
            except subprocess.CalledProcessError as e:
                print(f"[ASVspoof2021]   [WARNING] Download interrupted (Error code: {e.returncode}). Retrying in 30 seconds...")
                time.sleep(30) # 30초 대기 후 재시도

        if not success:
            print(f"[ASVspoof2021]   [ERROR] Failed after {max_retries} attempts.")
            print(f"[ASVspoof2021]   [TIP] 파일이 너무 커서 자꾸 끊긴다면, 웹 브라우저에서 직접 다운로드하여 '{target_dir}' 폴더에 압축을 풀어두는 것을 권장합니다.")
            return False
        # --- RETRY LOGIC END ---

    # 2. Organize structure (Handle Kaggle's extraction structure)
    print("\n[ASVspoof2021] [2/3] Checking file structure...")
    
    tar_files = list(target_dir.glob("ASVspoof2021_DF_eval_part*.tar.gz"))
    
    if not tar_files:
        subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            sub_tars = list(subdir.glob("ASVspoof2021_DF_eval_part*.tar.gz"))
            if sub_tars:
                print(f"[ASVspoof2021_DF]   Found tar files in subdir: {subdir.name}, moving them...")
                for f in sub_tars:
                    shutil.move(str(f), str(target_dir / f.name))
                tar_files = list(target_dir.glob("ASVspoof2021_DF_eval_part*.tar.gz"))
                break

    # 3. Extract inner tar.gz files
    flac_dir = target_dir / "flac"
    
    possible_flac_dirs = list(target_dir.glob("**/flac"))
    if possible_flac_dirs and not flac_dir.exists():
        print(f"[ASVspoof2021_DF]   Found existing 'flac' dir at {possible_flac_dirs[0]}, moving...")
        shutil.move(str(possible_flac_dirs[0]), str(flac_dir))

    if flac_dir.exists() and any(flac_dir.iterdir()):
         print(f"\n[ASVspoof2021_DF] [3/3] 'flac' directory is ready.")
    elif tar_files:
        print(f"\n[ASVspoof2021_DF] [3/3] Extracting inner tar.gz files ({len(tar_files)} files)...")
        for tar_path in sorted(tar_files):
            print(f"[ASVspoof2021_DF]   Extracting {tar_path.name}...")
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(target_dir)
            except Exception as e:
                print(f"[ASVspoof2021_DF]   [ERROR] Extraction failed for {tar_path.name}: {e}")
                return False
    else:
        print("[ASVspoof2021_DF]   [WARNING] No 'flac' folder and no 'tar.gz' files found.")

    # 4. Keys
    keys_dir = target_dir / "keys"
    if not keys_dir.exists():
        found_keys = list(target_dir.glob("**/keys"))
        if found_keys:
             shutil.move(str(found_keys[0]), str(keys_dir))
    
    if not keys_dir.exists():
        print("\n[ASVspoof2021_DF]   Keys not found in Kaggle download. Fetching from official source...")
        keys_dir.mkdir(parents=True, exist_ok=True)
        keys_url = "https://www.asvspoof.org/resources/DF-keys-full.tar.gz"
        keys_tar = keys_dir / "DF-keys-full.tar.gz"
        try:
            subprocess.run(["wget", "-c", "-O", str(keys_tar), keys_url], check=True)
            with tarfile.open(keys_tar, 'r:gz') as tar:
                tar.extractall(keys_dir)
            keys_tar.unlink()
        except Exception as e:
             print(f"[ASVspoof2021_DF]   [WARNING] Could not download keys: {e}")

    print("\n" + "=" * 70)
    print("[ASVspoof2021_DF] Process complete!")
    print("=" * 70)
    return True


def download_inthewild(data_dir: Path):
    """
    Download In-The-Wild dataset
    """
    print("=" * 70)
    print("[InTheWild] Starting download...")
    print("=" * 70)

    target_dir = data_dir / "InTheWild_hf"
    target_dir.mkdir(parents=True, exist_ok=True)

    download_url_hf = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
    filename = "release_in_the_wild.zip"
    output_path = target_dir / filename

    if output_path.exists():
        print(f"[InTheWild]   [SKIP] {filename} already exists")
    else:
        print(f"[InTheWild]   Downloading {filename}...")
        try:
            download_with_wget(download_url_hf, output_path)
        except Exception as e:
            print(f"[InTheWild]   [ERROR] Failed to download: {e}")
            return False

    print("\n[InTheWild]   Extracting...")
    if (target_dir / "release_in_the_wild").exists():
        print("[InTheWild]   [SKIP] Already extracted")
    else:
        try:
            with zipfile.ZipFile(output_path, 'r') as zf:
                zf.extractall(target_dir)
            print("[InTheWild]   [OK] Extraction complete")
        except Exception as e:
            print(f"[InTheWild]   [ERROR] Extraction failed: {e}")
            return False

    print("\n" + "=" * 70)
    print("[InTheWild] Download complete!")
    print("=" * 70)
    return True


def download_wavefake(data_dir: Path):
    """
    Download WaveFake dataset including Reference Data (LJSpeech, JSUT)
    """
    print("=" * 70)
    print("[WaveFake] Starting download (Generated Audio + References)...")
    print("=" * 70)

    target_dir = data_dir / "WaveFake"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success = True

    # ---------------------------------------------------------
    # 1. WaveFake Generated Audio
    # ---------------------------------------------------------
    print("\n[WaveFake] 1. Processing Generated Audio...")
    base_url = "https://zenodo.org/records/5642694/files"
    files = ["generated_audio.zip", "datasheet.pdf"]

    for filename in files:
        output_path = target_dir / filename
        if output_path.exists():
            print(f"[WaveFake]   [SKIP] {filename} already exists")
            continue

        print(f"[WaveFake]   Downloading {filename}...")
        try:
            download_with_wget(f"{base_url}/{filename}?download=1", output_path)
        except Exception as e:
            print(f"[WaveFake]   [ERROR] Failed to download {filename}: {e}")
            if filename == "generated_audio.zip":
                success = False

    # Extract Generated Audio
    if success:
        zip_file = target_dir / "generated_audio.zip"
        # Check if extracted (Simple check: look for one of the algo folders)
        if (target_dir / "generated_audio/ljspeech_melgan").exists():
            print(f"[WaveFake]   [SKIP] Generated audio seems already extracted")
        else:
            print("\n[WaveFake]   Extracting audio (~29GB)...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(target_dir)
                print("[WaveFake]   [OK] Generated audio extraction complete")
            except Exception as e:
                print(f"[WaveFake]   [ERROR] Extraction failed: {e}")
                success = False

    # ---------------------------------------------------------
    # 2. Reference: LJSpeech (Real)
    # ---------------------------------------------------------
    print("\n[WaveFake] 2. Processing Reference: LJSpeech...")
    ljs_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    ljs_filename = "LJSpeech-1.1.tar.bz2"
    ljs_path = target_dir / ljs_filename
    ljs_extract_dir = target_dir / "LJSpeech-1.1"

    if ljs_extract_dir.exists():
        print(f"[WaveFake]   [SKIP] LJSpeech directory found at {ljs_extract_dir}")
    else:
        # Download
        if not ljs_path.exists():
            print(f"[WaveFake]   Downloading {ljs_filename}...")
            try:
                download_with_wget(ljs_url, ljs_path)
            except Exception as e:
                print(f"[WaveFake]   [ERROR] LJSpeech download failed: {e}")
                success = False
        
        # Extract
        if ljs_path.exists():
            print(f"[WaveFake]   Extracting {ljs_filename}...")
            try:
                with tarfile.open(ljs_path, "r:bz2") as tar:
                    tar.extractall(target_dir)
                print(f"[WaveFake]   [OK] LJSpeech extracted")
            except Exception as e:
                print(f"[WaveFake]   [ERROR] LJSpeech extraction failed: {e}")
                success = False

    # ---------------------------------------------------------
    # 3. Reference: JSUT (Real)
    # ---------------------------------------------------------
    print("\n[WaveFake] 3. Processing Reference: JSUT...")
    # Direct link to avoid redirect issues
    jsut_url = "http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip" 
    jsut_filename = "jsut_ver1.1.zip"
    jsut_path = target_dir / jsut_filename
    jsut_extract_dir = target_dir / "jsut_ver1.1"

    if jsut_extract_dir.exists():
        print(f"[WaveFake]   [SKIP] JSUT directory found at {jsut_extract_dir}")
    else:
        # Download
        if not jsut_path.exists():
            print(f"[WaveFake]   Downloading {jsut_filename}...")
            try:
                download_with_wget(jsut_url, jsut_path)
            except Exception as e:
                print(f"[WaveFake]   [ERROR] JSUT download failed: {e}")
                success = False

        # Extract
        if jsut_path.exists():
            print(f"[WaveFake]   Extracting {jsut_filename}...")
            try:
                with zipfile.ZipFile(jsut_path, 'r') as zf:
                    zf.extractall(target_dir)
                print(f"[WaveFake]   [OK] JSUT extracted")
            except Exception as e:
                print(f"[WaveFake]   [ERROR] JSUT extraction failed: {e}")
                success = False

    print("\n" + "=" * 70)
    print(f"[WaveFake] Complete. Success: {success}")
    print("=" * 70)
    return success


# =========================================================================
# Parallel wrappers
# =========================================================================

def _download_asvspoof2021_wrapper(data_dir_str: str, result_queue: Queue):
    try:
        result = download_asvspoof2021_df(Path(data_dir_str))
        result_queue.put(("asvspoof2021", result))
    except Exception as e:
        print(f"[ASVspoof2021] Error: {e}")
        result_queue.put(("asvspoof2021", False))

def _download_inthewild_wrapper(data_dir_str: str, result_queue: Queue):
    try:
        result = download_inthewild(Path(data_dir_str))
        result_queue.put(("inthewild", result))
    except Exception as e:
        print(f"[InTheWild] Error: {e}")
        result_queue.put(("inthewild", False))

def _download_wavefake_wrapper(data_dir_str: str, result_queue: Queue):
    try:
        result = download_wavefake(Path(data_dir_str))
        result_queue.put(("wavefake", result))
    except Exception as e:
        print(f"[WaveFake] Error: {e}")
        result_queue.put(("wavefake", False))


def download_all_parallel(data_dir: Path, datasets: list):
    """Dataset-level parallel download"""
    print("=" * 70)
    print(f"Starting DATASET-LEVEL parallel download: {datasets}")
    print("=" * 70)
    
    result_queue = Queue()
    processes = []
    
    funcs = {
        "asvspoof2021": _download_asvspoof2021_wrapper,
        "inthewild": _download_inthewild_wrapper,
        "wavefake": _download_wavefake_wrapper,
    }

    for ds in datasets:
        if ds in funcs:
            p = Process(target=funcs[ds], args=(str(data_dir), result_queue))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    results = {}
    while not result_queue.empty():
        ds, ok = result_queue.get()
        results[ds] = ok
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Dataset Downloader")
    parser.add_argument("--dataset", "-d", required=True, 
                        choices=["all", "asvspoof2021", "inthewild", "wavefake"])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--parallel", "-p", action="store_true", 
                        help="Enable dataset-level parallelism")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data Dir: {data_dir}\n")

    if args.dataset == "all":
        targets = ["asvspoof2021", "inthewild", "wavefake"]
    else:
        targets = [args.dataset]

    if args.parallel and len(targets) > 1:
        results = download_all_parallel(data_dir, targets)
        success = all(results.values())
    else:
        success = True
        if "asvspoof2021" in targets:
            success &= download_asvspoof2021_df(data_dir)
        if "inthewild" in targets:
            success &= download_inthewild(data_dir)
        if "wavefake" in targets:
            success &= download_wavefake(data_dir)

    if success:
        print("\nSUCCESS: All tasks finished.")
    else:
        print("\nFAILURE: Some tasks failed.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())