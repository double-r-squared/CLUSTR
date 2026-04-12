#!/usr/bin/env bash
# deploy.sh — Bootstrap or update a remote CLUSTR worker node.
#
# Usage:
#   ./scripts/deploy.sh [path/to/clustr.conf] [--clean]
#
#   --clean   Kill any running worker, clear logs and work dir before deploying.
#
# Requires: scp, ssh, sudo access on the remote machine.
# The remote machine must have g++ installed (apt install build-essential).

set -euo pipefail

CONF="${1:-clustr.conf}"
CLEAN=0
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN=1
done

# ---------------------------------------------------------------
# Parse config file
# ---------------------------------------------------------------

parse_conf() {
    local key="$1"
    local default="${2:-}"
    local val
    val=$(grep -E "^${key}\s*=" "$CONF" 2>/dev/null \
          | head -1 | sed 's/^[^=]*=//; s/#.*//' | xargs)
    echo "${val:-$default}"
}

SCHEDULER_IP=$(parse_conf "scheduler_ip" "")
SCHEDULER_PORT=$(parse_conf "scheduler_port" "9999")
DEPLOY_USER=$(parse_conf "deploy_user" "")
DEPLOY_HOST=$(parse_conf "deploy_host" "")
SSH_KEY=$(parse_conf "ssh_key_path" "$HOME/.ssh/id_rsa")
REMOTE_INSTALL_CFG=$(parse_conf "remote_install_path" "")
WORK_DIR=$(parse_conf "work_dir" "/tmp/clustr")

SSH_KEY="${SSH_KEY/#\~/$HOME}"

# ---------------------------------------------------------------
# Validate
# ---------------------------------------------------------------

if [[ -z "$DEPLOY_USER" || -z "$DEPLOY_HOST" ]]; then
    echo "ERROR: deploy_user and deploy_host must be set in $CONF" >&2
    exit 1
fi

if [[ -z "$SCHEDULER_IP" ]]; then
    echo "ERROR: scheduler_ip must be set in $CONF" >&2
    exit 1
fi

ASIO_INCLUDE_DIR=$(find ./build/_deps -name "asio.hpp" -exec dirname {} \; 2>/dev/null | head -1)
if [[ -z "$ASIO_INCLUDE_DIR" ]]; then
    echo "ERROR: ASIO headers not found in build/_deps." >&2
    echo "       Run: cmake -B build && cmake --build build/ first" >&2
    exit 1
fi
echo "  ASIO:      $ASIO_INCLUDE_DIR"

REMOTE="${DEPLOY_USER}@${DEPLOY_HOST}"

SSH_OPTS="-o StrictHostKeyChecking=no"
if [[ -f "$SSH_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

REMOTE_INSTALL="${REMOTE_INSTALL_CFG:-/var/tmp/clustr_worker}"
WORKER_CONF="/var/tmp/clustr/worker.conf"

echo "CLUSTR Deploy"
echo "  Remote:    $REMOTE"
echo "  Install:   $REMOTE_INSTALL"
echo "  Scheduler: $SCHEDULER_IP:$SCHEDULER_PORT"
echo "  work_dir:  $WORK_DIR"
echo ""

# Unique ID for this deploy run — prevents collisions when deploying to the
# same host in parallel (e.g. "Deploy All" from the TUI).
DEPLOY_ID="${DEPLOY_HOST}_$(date +%s)_$$"

# Remote temp paths — all scoped to this deploy run
REMOTE_TAR="/tmp/clustr_src_${DEPLOY_ID}.tar.gz"
REMOTE_DIR="/tmp/clustr_src_${DEPLOY_ID}"
REMOTE_BIN="/tmp/clustr_worker_${DEPLOY_ID}"

SRC_TARBALL=""
trap '[[ -n "$SRC_TARBALL" ]] && rm -f "$SRC_TARBALL"' EXIT

# ---------------------------------------------------------------
# Step 1: Bundle source code locally
# ---------------------------------------------------------------

echo "[1/5] Bundling worker source..."

BUNDLE_DIR=$(mktemp -d)
mkdir -p "$BUNDLE_DIR/include" "$BUNDLE_DIR/src"

cp -r include/.             "$BUNDLE_DIR/include/"
cp src/capability_detector.cpp \
   src/file_transfer.cpp \
   src/protocol.cpp \
   src/process_monitor.cpp \
   src/remote_exec.cpp \
   src/tcp_server.cpp       "$BUNDLE_DIR/src/"
cp client/client.cpp        "$BUNDLE_DIR/"

cp -r "$ASIO_INCLUDE_DIR"   "$BUNDLE_DIR/asio_include"

# Use mktemp for the directory only; build the tarball path from it
SRC_TARBALL=$(mktemp -t clustr_deploy)
# mktemp creates a plain file; we want a .tar.gz — rename it
mv "$SRC_TARBALL" "${SRC_TARBALL}.tar.gz"
SRC_TARBALL="${SRC_TARBALL}.tar.gz"

tar -czf "$SRC_TARBALL" -C "$BUNDLE_DIR" .
rm -rf "$BUNDLE_DIR"

echo "    Tarball: $(du -sh "$SRC_TARBALL" | cut -f1)"

# ---------------------------------------------------------------
# Step 2: Upload
# ---------------------------------------------------------------

echo "[2/5] Uploading to $REMOTE..."
scp $SSH_OPTS "$SRC_TARBALL" "$REMOTE:$REMOTE_TAR"

# ---------------------------------------------------------------
# Step 3: Compile on remote
# ---------------------------------------------------------------

echo "[3/5] Compiling on $REMOTE (this may take a minute)..."
ssh $SSH_OPTS "$REMOTE" bash -s << COMPILE_SCRIPT
set -e
mkdir -p "$REMOTE_DIR"
tar -xzf "$REMOTE_TAR" -C "$REMOTE_DIR"
cd "$REMOTE_DIR"

if ! command -v g++ &> /dev/null; then
    echo "  Installing build tools..."
    sudo apt-get update -qq
    sudo apt-get install -y build-essential < /dev/null
fi

echo "  Compiling worker..."
g++ -std=c++20 -O2 \
    -DASIO_STANDALONE -DASIO_NO_DEPRECATED \
    -I./include -I./asio_include \
    src/capability_detector.cpp \
    src/file_transfer.cpp \
    src/protocol.cpp \
    src/process_monitor.cpp \
    src/remote_exec.cpp \
    src/tcp_server.cpp \
    client.cpp \
    -o "$REMOTE_BIN" \
    -lpthread

echo "  Compilation successful!"

# Copy headers to a stable location so MPI job files can include them.
# /var/tmp survives reboots and won't be clobbered by a subsequent deploy.
echo "  Installing job headers to /var/tmp/clustr/include..."
mkdir -p /var/tmp/clustr/include /var/tmp/clustr/asio_include
cp -r "$REMOTE_DIR/include/."       /var/tmp/clustr/include/
cp -r "$REMOTE_DIR/asio_include/."  /var/tmp/clustr/asio_include/
echo "  Headers installed."

# Clean up deploy-specific temp files
rm -rf "$REMOTE_DIR" "$REMOTE_TAR"
COMPILE_SCRIPT

# ---------------------------------------------------------------
# Step 4: Install binary
# ---------------------------------------------------------------

echo "[4/5] Installing worker binary..."
ssh $SSH_OPTS "$REMOTE" bash -s << INSTALL_SCRIPT
set -e
mv "$REMOTE_BIN" "$REMOTE_INSTALL"
chmod +x "$REMOTE_INSTALL"
echo "  Installed to $REMOTE_INSTALL"
INSTALL_SCRIPT

# ---------------------------------------------------------------
# Step 5: Configure and start worker
# ---------------------------------------------------------------

echo "[5/5] Configuring and starting worker..."

CONF_CONTENT=$(cat <<EOF
scheduler_ip=$SCHEDULER_IP
scheduler_port=$SCHEDULER_PORT
work_dir=$WORK_DIR
EOF
)

ssh $SSH_OPTS "$REMOTE" bash -s << REMOTE_SCRIPT
set -e
mkdir -p "$(dirname "$WORKER_CONF")" "$WORK_DIR"
cat > $WORKER_CONF <<'CONF_EOF'
$CONF_CONTENT
CONF_EOF

pkill -f '$REMOTE_INSTALL' 2>/dev/null || true
sleep 0.5

nohup $REMOTE_INSTALL $WORKER_CONF \
    </dev/null >> /tmp/clustr_worker.log 2>&1 &
disown || true
sleep 1

if pgrep -f '$REMOTE_INSTALL' > /dev/null; then
    echo "  [OK] Worker started"
else
    echo "  [FAIL] Worker failed - check /tmp/clustr_worker.log"
    tail -20 /tmp/clustr_worker.log
    exit 1
fi
REMOTE_SCRIPT

echo ""
echo "[OK] Deployment complete!"
echo ""
echo "View worker logs:"
echo "  ssh $REMOTE 'tail -f /tmp/clustr_worker.log'"
echo ""
echo "Stop worker:"
echo "  ssh $REMOTE 'pkill -f $REMOTE_INSTALL'"