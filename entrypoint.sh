#!/bin/bash
# This script is the gatekeeper of our dojo, ensuring all is fresh and ready!
set -e # Exit immediately if a command exits with a non-zero status - no room for slacking!

echo "🥋 Greetings from the entrypoint! Checking for wisdom updates from the ancient repo..."
git fetch origin # Quietly peek at the remote scrolls.

# Compare our local scroll versions with the master scrolls on the remote mountain (origin/main).
LOCAL=$(git rev-parse HEAD)         # Our current knowledge level.
REMOTE=$(git rev-parse origin/main) # The latest wisdom available.

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "🎯 Scrolls of new wisdom found! Pulling them down from the mountain..."
    git pull origin main # Acquire the new knowledge!

    echo "🔧 The new scrolls might require new tools! Updating dependencies..."
    . .venv/bin/activate # Step into the virtual dojo.
    # The original had `uv pip install -r pyproject.toml`, but if pyproject.toml changes, `uv pip install .` is also great!
    # For consistency with Dockerfile, let's assume pyproject.toml is the source of truth for deps.
    uv pip install -r pyproject.toml # Sharpen our tools (dependencies).
else
    echo "✅ All scrolls are current. Our wisdom is already at its peak (for now)!"
fi

# All checks complete! Now, let's begin the actual training (run the Python script).
# Handing over control to the grandmaster Python script, passing along any ancient commands (arguments).
echo "🚀 Launching the grandmaster training script: train.py!"
exec ./.venv/bin/python train.py "$@"
