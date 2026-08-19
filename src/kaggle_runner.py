"""
Automated Kaggle Dual-T4 GPU Dispatch & Output Extractor.
Seamlessly compiles the modular codebase and triggers Kaggle cloud execution.
"""
import os
import sys
import time
import subprocess

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def main():
    kernel_slug = "aliazizi1/predicting-smartphone-addiction-elite"
    print("=" * 60)
    print("🚀 [Kaggle 2x T4 GPU Runner] Initiating Cloud Dispatch...")
    print("=" * 60)

    # 1. Compile clean modular codebase into Kaggle notebook
    print("📦 Step 1: Compiling local modular source into standalone notebook...")
    ret, out, err = run_command("python3 src/compile_notebook.py")
    if ret != 0:
        print(f"❌ Compilation failed: {err}")
        sys.exit(1)
    print("✅ Notebook compilation successful: predicting-smartphone-addiction-elite.ipynb")

    # 2. Push to Kaggle Cloud with 2x T4 GPU
    print(f"☁️ Step 2: Pushing kernel to Kaggle Cloud GPU -> {kernel_slug}...")
    ret, out, err = run_command("kaggle kernels push -p .")
    if ret != 0:
        print(f"❌ Push failed: {err}")
        sys.exit(1)
    print(f"✅ Kernel pushed successfully:\n{out}")

    print("\n⏳ Step 3: Cloud GPU worker launched. To monitor or fetch outputs anytime:")
    print(f"   kaggle kernels status {kernel_slug}")
    print(f"   kaggle kernels output {kernel_slug} -p outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()
