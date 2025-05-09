FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Greetings, traveler! Setting up the foundational OS scrolls (dependencies).
# We're grabbing Python, pip for its magic, git for version-sorcery, and curl for fetching ancient artifacts.
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/* # Tidying up, because a clean dojo is a happy dojo.

# Now, let's summon 'uv', the swift package manager from the Astral plane.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make sure 'uv' is on the PATH, so its powers are readily available.
ENV PATH="/root/.cargo/bin:${PATH}"

# Establishing our sacred training ground: the /app directory.
WORKDIR /app

# Copying all your precious code and configurations into our training ground.
# The .dockerignore talisman ensures we only bring what's needed for the quest!
COPY . .

# Designating a special scroll-cache (Huggingface cache) on a volume, so it persists across training sessions.
ENV HF_HOME=/app/.cache/huggingface

# Installing dependencies with the mighty 'uv'.
# First, create a virtual dojo (.venv), then activate it,
# upgrade pip (just in case it's feeling old), and finally, install our project's needs.
RUN uv venv && . .venv/bin/activate && \
    uv pip install -U pip && \
    uv pip install . # Master Lonn-San's wisdom: 'uv pip install .' reads pyproject.toml directly!

# Adding the entrypoint script - our dojo's master of ceremonies.
COPY entrypoint.sh /entrypoint.sh
# Granting it the power to execute (chmod +x).
RUN chmod +x /entrypoint.sh

# And now, the grand entrance! The entrypoint script takes the stage.
ENTRYPOINT ["/entrypoint.sh"]
