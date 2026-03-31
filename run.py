import subprocess
import sys
import os
import time
import signal

def run_command(command, cwd, label):
    print(f"[*] Starting {label}...")
    return subprocess.Popen(command, cwd=cwd, shell=True)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "Code", "backend")
    frontend_dir = os.path.join(base_dir, "Code", "frontend")

    # 1. Start Backend (ensure venv is used)
    # On Mac/Linux, venv is in Code/venv
    venv_python = os.path.join(base_dir, "Code", "venv", "bin", "python3")
    if not os.path.exists(venv_python):
        venv_python = "python3" # Fallback
    
    backend_cmd = f"{venv_python} main.py"
    backend_proc = run_command(backend_cmd, backend_dir, "Backend (FastAPI)")

    # 2. Start Frontend
    frontend_cmd = "npm run dev"
    frontend_proc = run_command(frontend_cmd, frontend_dir, "Frontend (Next.js)")

    print("\n[!] DeepShield AI is starting up...")
    print("[!] Backend: http://localhost:8000")
    print("[!] Frontend: http://localhost:3000")
    print("[!] Press Ctrl+C to stop both services.\n")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("[!] Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("[!] Frontend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n[!] Stopping services...")
        backend_proc.send_signal(signal.SIGINT)
        frontend_proc.send_signal(signal.SIGINT)
        backend_proc.wait()
        frontend_proc.wait()
        print("[!] Cleanup complete.")

if __name__ == "__main__":
    main()
