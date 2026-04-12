#!/usr/bin/env bash
# setup_worker.sh — One-time worker node setup. Run this ONCE per cluster
# from your local terminal before deploying via the TUI.
#
# Usage:
#   ./scripts/setup_worker.sh [path/to/system.conf]
#
# What it does:
#   1. Generates an SSH key pair locally (skipped if one already exists)
#   2. Copies the public key to each worker (password prompted once per machine)
#   3. Configures passwordless sudo on each worker (sudo password prompted once)
#   4. Installs build tools (g++) on each worker if missing
#
# After this script completes, all future deploys run fully non-interactive.

set -euo pipefail

CONF="${1:-system.conf}"

parse_conf() {
    local key="$1" default="${2:-}"
    local val
    val=$(grep -E "^${key}\s*=" "$CONF" 2>/dev/null \
          | head -1 | sed 's/^[^=]*=//; s/#.*//' | xargs)
    echo "${val:-$default}"
}

SSH_KEY=$(parse_conf "ssh_key_path" "$HOME/.ssh/id_rsa")
SSH_KEY="${SSH_KEY/#\~/$HOME}"
SSH_PUB="${SSH_KEY}.pub"

# ---------------------------------------------------------------
# Step 1: Generate SSH key pair (local)
# ---------------------------------------------------------------

echo ""
echo "=== SSH Key ==="
if [[ -f "$SSH_KEY" ]]; then
    echo "  [skip] Key already exists: $SSH_KEY"
else
    echo "  Generating ed25519 key pair at $SSH_KEY ..."
    echo "  (press Enter three times to accept defaults)"
    echo ""
    ssh-keygen -t ed25519 -f "$SSH_KEY"
    echo ""
    echo "  [ok]   Key generated"
fi

# ---------------------------------------------------------------
# Step 2+: Per-worker setup
# ---------------------------------------------------------------

# Collect all worker sections.
# grep exits 1 when there are no matches, which would kill the script under
# set -e, so we capture into a variable with || true first, then feed it
# via here-string to avoid process substitution issues on bash 3.2 (macOS).
WORKERS=()
_sections=$(grep -E '^\[.+\]' "$CONF" 2>/dev/null | tr -d '[]' || true)
if [[ -n "$_sections" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && WORKERS+=("$line")
    done <<< "$_sections"
fi

if [[ ${#WORKERS[@]} -eq 0 ]]; then
    echo ""
    echo "ERROR: No worker sections found in $CONF" >&2
    exit 1
fi

for WORKER in "${WORKERS[@]}"; do
    SECTION=$(awk "/^\[$WORKER\]/{found=1; next} found && /^\[/{exit} found{print}" "$CONF" || true)
    get_field() {
        local key="$1" default="${2:-}"
        local val
        val=$(echo "$SECTION" | grep -E "^${key}\s*=" \
              | head -1 | sed 's/^[^=]*=//; s/#.*//' | xargs)
        if [[ -z "$val" ]]; then
            val=$(parse_conf "$key" "$default")
        fi
        echo "$val"
    }

    DEPLOY_HOST=$(get_field "deploy_host" "")
    DEPLOY_USER=$(get_field "deploy_user" "")

    if [[ -z "$DEPLOY_HOST" || -z "$DEPLOY_USER" ]]; then
        echo ""
        echo "[$WORKER] Skipping — missing deploy_host or deploy_user"
        continue
    fi

    REMOTE="${DEPLOY_USER}@${DEPLOY_HOST}"

    echo ""
    echo "=== $WORKER ($REMOTE) ==="

    # Step 2: Copy public key (uses password auth — prompted once)
    echo "  [2/3] Copying public key (enter $DEPLOY_USER's login password when prompted)..."
    ssh-copy-id -i "$SSH_PUB" -o StrictHostKeyChecking=no "$REMOTE"
    echo "  [ok]  Key installed — future connections are passwordless"

    # Step 3: Remote setup over key auth (sudo -S reads password from stdin — no PTY needed)
    echo "  [3/3] Configuring remote..."
    printf "  Enter sudo password for %s on %s (%s): " "$DEPLOY_USER" "$WORKER" "$DEPLOY_HOST"
    read -rs SUDO_PASS
    echo ""
    SSH_OPTS="-o StrictHostKeyChecking=no -i $SSH_KEY"
    ssh $SSH_OPTS "$REMOTE" bash -s << SETUP
set -e
SPASS='$SUDO_PASS'

# Passwordless sudo — write content to a temp file first (no sudo needed),
# then sudo-copy it so stdin is only ever used for the password.
SUDOERS_LINE="$DEPLOY_USER ALL=(ALL) NOPASSWD: ALL"
TMPFILE=\$(mktemp)
echo "\$SUDOERS_LINE" > "\$TMPFILE"

if printf '%s\n' "\$SPASS" | sudo -S grep -q "NOPASSWD" /etc/sudoers.d/clustr 2>/dev/null; then
    echo "  [skip] NOPASSWD already configured"
    rm -f "\$TMPFILE"
else
    printf '%s\n' "\$SPASS" | sudo -S cp "\$TMPFILE" /etc/sudoers.d/clustr
    printf '%s\n' "\$SPASS" | sudo -S chmod 0440 /etc/sudoers.d/clustr
    rm -f "\$TMPFILE"
    echo "  [ok]   NOPASSWD sudo configured — verifying..."
    printf '%s\n' "\$SPASS" | sudo -S grep "NOPASSWD" /etc/sudoers.d/clustr
fi

# Build tools
if command -v g++ &>/dev/null; then
    echo "  [skip] g++ already installed: \$(g++ --version | head -1)"
else
    echo "  Installing build tools..."
    printf '%s\n' "\$SPASS" | sudo -S apt-get update -qq
    printf '%s\n' "\$SPASS" | sudo -S apt-get install -y build-essential < /dev/null
    echo "  [ok]   g++ installed"
fi

echo "  Setup complete for $WORKER"
SETUP

done

echo ""
echo "All workers configured. You can now deploy from the TUI."
