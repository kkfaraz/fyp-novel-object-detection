#!/bin/bash
# Stop script for the Cooperative NOD backend server
# It finds any python process running server.py on port 8000 and kills it gracefully

echo "Stopping backend server on port 8000..."

# Find PIDs listening on port 8000
PIDS=$(lsof -t -i :8000)

if [ -z "$PIDS" ]; then
  echo "✅ No server is currently running on port 8000."
else
  echo "Found server running with PID: $PIDS"
  kill $PIDS
  echo "✅ Server stopped successfully."
fi
