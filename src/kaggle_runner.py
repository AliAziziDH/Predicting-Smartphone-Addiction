"""
Automated Kaggle GPU Dispatcher, Poller & Auto-Submitter.
Compiles modular codebase, dispatches to Kaggle GPU, monitors status,
and automatically submits the generated submission.csv to the competition immediately upon completion.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and v:
                    os.environ[k] = v


def ensure_kaggle_credentials():
    load_env()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    if key:
        os.environ["KAGGLE_API_TOKEN"] = key
        (kaggle_dir / "access_token").write_text(key)

    kaggle_json = kaggle_dir / "kaggle.json"

    if username and key and username != "your_kaggle_username" and key != "your_kaggle_key":
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        try:
            kaggle_json.chmod(0o600)
        except Exception:
            pass
        return username

    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            u = data.get("username", "").strip()
            if u:
                os.environ["KAGGLE_USERNAME"] = u
                os.environ["KAGGLE_KEY"] = data.get("key", "").strip()
                return u
        except Exception:
            pass

    return username or None


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def main():
    print("=" * 75, flush=True)
    print("🚀 [Kaggle Cloud GPU Dispatch & Immediate Auto-Submitter]", flush=True)
    print("=" * 75, flush=True)

    username = ensure_kaggle_credentials()
    if not username:
        print("❌ Error: Kaggle credentials not found in .env or ~/.kaggle/kaggle.json", flush=True)
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file.", flush=True)
        sys.exit(1)

    kernel_slug = f"{username}/predicting-smartphone-addiction-elite"

    # Update kernel-metadata.json
    meta_path = Path("kernel-metadata.json")
    meta = {
        "id": kernel_slug,
        "title": "predicting-smartphone-addiction-elite",
        "code_file": "predicting-smartphone-addiction-elite.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": ["playground-series-s6e8"],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Configured kernel metadata for user: {username} -> {kernel_slug}", flush=True)

    # 1. Compile clean modular codebase into Kaggle standalone notebook
    print("\n📦 Step 1: Compiling modular codebase into standalone GPU notebook...", flush=True)
    ret, out, err = run_cmd("python src/compile_notebook.py")
    if ret != 0:
        print(f"❌ Notebook compilation failed: {err}")
        sys.exit(1)
    print("✅ Notebook compiled: predicting-smartphone-addiction-elite.ipynb")

    # 2. Push to Kaggle Cloud GPU
    print(f"\n☁️ Step 2: Pushing kernel to Kaggle Cloud GPU -> {kernel_slug}...")
    ret, out, err = run_cmd("kaggle kernels push -p .")
    if ret != 0:
        print(f"❌ Kaggle push failed:\nSTDOUT: {out}\nSTDERR: {err}")
        sys.exit(1)
    print(f"✅ Kernel pushed successfully to Kaggle GPU worker!\n{out}")

    # 3. Monitor Status & Auto-Submit
    print(f"\n⏳ Step 3: Polling Kaggle GPU execution status for {kernel_slug}...")
    start_time = time.time()
    poll_interval = 20

    while True:
        time.sleep(poll_interval)
        elapsed = int(time.time() - start_time)
        ret, out, err = run_cmd(f"kaggle kernels status {kernel_slug}")
        if ret != 0:
            print(f"⚠️ [Status Check Error ({elapsed}s)]: {err}")
            continue

        print(f"⏱️ [{elapsed//60}m {elapsed%60}s elapsed] Status: {out}")

        status_lower = out.lower()
        if "complete" in status_lower:
            print("\n🎉 Kaggle GPU Execution COMPLETE!")
            break
        elif "error" in status_lower or "cancel" in status_lower:
            print(f"\n❌ Execution finished with error/cancelled status: {out}")
            run_cmd(f"kaggle kernels output {kernel_slug} -p outputs/")
            sys.exit(1)

    # 4. Download output and submit immediately
    print("\n📥 Step 4: Fetching generated submission artifact from Kaggle...")
    os.makedirs("outputs", exist_ok=True)
    ret, out, err = run_cmd(f"kaggle kernels output {kernel_slug} -p outputs/")
    print(f"Output download: {out}")

    sub_file = "outputs/submission.csv"
    if not os.path.exists(sub_file):
        # Check current directory
        if os.path.exists("submission.csv"):
            sub_file = "submission.csv"

    if os.path.exists(sub_file):
        print(f"\n🚀 Step 5: Submitting {sub_file} immediately to playground-series-s6e8...")
        ret, out, err = run_cmd(f'kaggle competitions submit -c playground-series-s6e8 -f "{sub_file}" -m "Elite 10-Fold 4-Way GPU Ensemble (LGB+XGB+CAT+NN + Stacker)"')
        print(f"Submission Response:\n{out}")
        if err:
            print(f"Notice: {err}")

        # Check leaderboard
        time.sleep(10)
        print("\n📊 Step 6: Fetching submission score from leaderboard...")
        ret, out, err = run_cmd("kaggle competitions submissions -c playground-series-s6e8")
        if ret == 0:
            print(f"Recent Submissions:\n{out[:800]}")
    else:
        print(f"⚠️ Could not find submission file at {sub_file}. Please check outputs/ directory.")

    print("\n" + "=" * 75)
    print("🏁 Pipeline finished.")
    print("=" * 75)


if __name__ == "__main__":
    main()
