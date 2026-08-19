#!/bin/bash
# Autonomous Multi-Submission Background Launcher
mkdir -p logs
echo "🚀 Launching Autonomous Offline Daemon for 4 Submissions..."
nohup python3 src/autonomous_daemon.py 4 > logs/autonomous_daemon.log 2>&1 &
echo "✅ Daemon PID: $!"
echo "📄 Logs streaming to: logs/autonomous_daemon.log"
