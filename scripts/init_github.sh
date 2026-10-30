#!/usr/bin/env bash
# =============================================================================
# Autonomous Graph-RAG Agent — GitHub Repository Initialization & Push Script
# Target Repository: Sawy05/autonomous-graph-rag-agent
# =============================================================================
set -euo pipefail

REPO_SLUG="Sawy05/autonomous-graph-rag-agent"
DEFAULT_BRANCH="main"

echo "=================================================================="
echo " Initializing GitHub Repository: ${REPO_SLUG}"
echo "=================================================================="

# 1. Verify Git is installed
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] 'git' is not installed or not on PATH." >&2
    exit 1
fi

# 2. Verify GitHub CLI (gh) is installed and authenticated
if ! command -v gh >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI ('gh') is not installed. Install it from https://cli.github.com/" >&2
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI is not authenticated. Run 'gh auth login' first." >&2
    exit 1
fi

# 3. Initialize local Git repository with default branch 'main'
if [ ! -d ".git" ]; then
    echo "[INFO] Initializing local Git repository on branch '${DEFAULT_BRANCH}'..."
    git init -b "${DEFAULT_BRANCH}"
else
    echo "[INFO] Existing Git repository detected. Ensuring branch '${DEFAULT_BRANCH}'..."
    git checkout -B "${DEFAULT_BRANCH}"
fi

# 4. Stage all project files and create initial commit
echo "[INFO] Staging all repository files..."
git add -A

if git diff --cached --quiet; then
    echo "[INFO] Working tree clean; no new changes to commit."
else
    echo "[INFO] Creating initial production commit..."
    git commit -m "feat: initial production release of Autonomous Graph-RAG Agent microservice"
fi

# 5. Create public GitHub repository and push to origin main
if gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
    echo "[INFO] Remote repository '${REPO_SLUG}' already exists on GitHub."
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/${REPO_SLUG}.git"
    fi
    echo "[INFO] Pushing branch '${DEFAULT_BRANCH}' to origin..."
    git push -u origin "${DEFAULT_BRANCH}"
else
    echo "[INFO] Creating public GitHub repository '${REPO_SLUG}' and pushing '${DEFAULT_BRANCH}'..."
    gh repo create "${REPO_SLUG}" \
        --public \
        --source=. \
        --remote=origin \
        --description "Production-grade, self-correcting Enterprise Graph-RAG microservice with LangGraph, Qdrant, BM25, Cross-Encoder Reranking, and Neo4j" \
        --push
fi

echo "=================================================================="
echo " Successfully published to https://github.com/${REPO_SLUG}"
echo "=================================================================="
