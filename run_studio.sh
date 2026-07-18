#!/bin/bash

# 1. Compile the HTML book to ensure all recent changes are baked in
echo "=== Recompiling study guide chapters ==="
python3 compile_book.py

# 2. Find an available port starting from 8000
PORT=8000
while ss -tln | grep -q ":$PORT " &>/dev/null; do
    echo "Port $PORT is currently in use, trying $((PORT+1))..."
    PORT=$((PORT+1))
done

echo "=== Starting HTTP Server on port $PORT ==="
# 3. Start Python's built-in web server in the background
# We serve the root directory so both the split and monolithic HTML books are accessible
python3 -m http.server $PORT &
SERVER_PID=$!

# Cleanup handler to kill the background server when the script is stopped
cleanup() {
    echo ""
    echo "=== Stopping HTTP Server (PID $SERVER_PID) ==="
    kill $SERVER_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Wait briefly for the server socket to open
sleep 0.5

# 5. Open the HTML Book in the default browser (using URL encoding for spaces)
URL="http://localhost:$PORT/HTML%20BOOK/index.html"
echo "=== Launching Book in default browser ==="
echo "Opening URL: $URL"

if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" &>/dev/null &
elif command -v sensible-browser &>/dev/null; then
    sensible-browser "$URL" &>/dev/null &
elif command -v google-chrome &>/dev/null; then
    google-chrome "$URL" &>/dev/null &
elif command -v firefox &>/dev/null; then
    firefox "$URL" &>/dev/null &
else
    echo "Warning: Could not automatically open the browser."
    echo "Please open your browser manually and visit: $URL"
fi

# Keep script alive to keep background server running
echo "Press [Ctrl+C] to stop the local server."
wait $SERVER_PID
