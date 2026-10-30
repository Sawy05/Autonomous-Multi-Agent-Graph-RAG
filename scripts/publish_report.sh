#!/usr/bin/env bash
# =============================================================================
# Autonomous Graph-RAG Agent — Thesis & Documentation Publishing Pipeline
# Author: Ahmed Elsawy (@Sawy05)
# Target Repository: https://github.com/Sawy05/autonomous-graph-rag-agent
# =============================================================================
set -euo pipefail

REPO_OWNER="Sawy05"
REPO_NAME="autonomous-graph-rag-agent"
TARGET_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}"
REPORT_DOC_URL="${TARGET_URL}/blob/main/docs/report/THESIS.md"

echo "=================================================================="
echo " Publishing Thesis & System Documentation to GitHub"
echo " Target: ${TARGET_URL}"
echo "=================================================================="

# 1. Validate git installation
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] 'git' is not installed or not available in PATH." >&2
    exit 1
fi

# 2. Check for GitHub CLI (gh)
GH_AVAILABLE=true
if ! command -v gh >/dev/null 2>&1; then
    echo "[WARN] GitHub CLI ('gh') is not installed. Proceeding with native git remote pipeline."
    GH_AVAILABLE=false
else
    if ! gh auth status >/dev/null 2>&1; then
        echo "[WARN] GitHub CLI ('gh') is not authenticated. Proceeding with configured git credentials."
        GH_AVAILABLE=false
    fi
fi

# 3. Ensure git branch is main
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
if [ "${CURRENT_BRANCH}" != "main" ]; then
    echo "[INFO] Switching to branch 'main'..."
    git checkout -B main
fi

# 4. Configure or verify remote origin
EXISTING_ORIGIN="$(git config --get remote.origin.url || true)"
if [ -z "${EXISTING_ORIGIN}" ]; then
    echo "[INFO] Setting remote origin to ${TARGET_URL}.git..."
    git remote add origin "${TARGET_URL}.git"
else
    echo "[INFO] Verified existing remote origin: ${EXISTING_ORIGIN}"
fi

# 5. Verify .gitignore protects LaTeX artifacts while tracking docs/report/
echo "[INFO] Verifying .gitignore rules for LaTeX and Markdown docs..."
git check-ignore docs/report/THESIS.md >/dev/null 2>&1 && {
    echo "[ERROR] docs/report/THESIS.md is erroneously ignored by .gitignore!" >&2
    exit 1
} || true

git check-ignore docs/report/latex/main.tex >/dev/null 2>&1 && {
    echo "[ERROR] docs/report/latex/main.tex is erroneously ignored by .gitignore!" >&2
    exit 1
} || true

# 6. Stage documentation and workflow files
echo "[INFO] Staging thesis, LaTeX sources, CI workflow, and publishing scripts..."
git add docs/report/THESIS.md
git add docs/report/latex/
git add .github/workflows/compile-report.yml
git add .gitignore
git add scripts/publish_report.sh

# 7. Create conventional commit if changes exist
if git diff --cached --quiet; then
    echo "[INFO] No unstaged changes detected. Working tree clean."
else
    echo "[INFO] Committing documentation suite..."
    git commit -m "docs: publish comprehensive thesis, system proposal, and CI compilation workflow"
fi

# 8. Push to GitHub remote
echo "[INFO] Attempting push to origin main..."
PUSH_SUCCESS=false
if git push -u origin main 2>/dev/null; then
    PUSH_SUCCESS=true
elif [ "${GH_AVAILABLE}" = true ]; then
    echo "[INFO] Standard push returned non-zero. Attempting push via gh repository helper..."
    if gh repo view "${REPO_OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
        git push -u origin main && PUSH_SUCCESS=true || true
    else
        echo "[INFO] Creating remote repository ${REPO_OWNER}/${REPO_NAME} on GitHub..."
        gh repo create "${REPO_OWNER}/${REPO_NAME}" --public --source=. --remote=origin --push && PUSH_SUCCESS=true || true
    fi
fi

echo "=================================================================="
if [ "${PUSH_SUCCESS}" = true ]; then
    echo " SUCCESS: All documentation successfully committed and pushed!"
    echo " View Online Thesis:"
    echo "   ${REPORT_DOC_URL}"
    echo " LaTeX Automated CI PDF Action:"
    echo "   ${TARGET_URL}/actions/workflows/compile-report.yml"
else
    echo " LOCAL COMMITS COMPLETE: Git commit recorded on branch 'main'."
    echo " Remote push requires credentials or GitHub CLI authentication."
    echo " Run 'gh auth login' or configure your SSH/PAT, then run:"
    echo "   git push -u origin main"
    echo ""
    echo " Direct target URLs after push:"
    echo "   Documentation Report: ${REPORT_DOC_URL}"
    echo "   Repository:           ${TARGET_URL}"
fi
echo "=================================================================="
