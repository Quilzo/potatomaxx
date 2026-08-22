#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Replace an email address throughout this repository's commit history.
#
# Git stores the author and committer address in every commit object, so commits
# made before an identity was corrected still carry the old one, and forges
# display it publicly. Fixing that means rewriting those commits, which changes
# their hashes and therefore requires a force-push.
#
# Usage:
#   ./FIX-GIT-HISTORY.sh OLD_EMAIL NEW_EMAIL [NEW_NAME]
#
# The addresses are arguments rather than constants on purpose: hardcoding the
# address you are trying to remove would publish it in the very file meant to
# remove it. (That was the first version of this script.)
#
# Safe when nobody holds a clone whose history the rewrite would invalidate.
# Check before running:
#
#   gh repo view OWNER/REPO --json forkCount,stargazerCount,watchers
#
# Run from the repository root. It is a script you execute yourself rather than
# something run on your behalf, because force-pushing over published history is
# not a decision to delegate.
set -eu

if [ $# -lt 2 ]; then
    echo "usage: $0 OLD_EMAIL NEW_EMAIL [NEW_NAME]" >&2
    exit 2
fi
OLD=$1
NEW=$2
NAME=${3:-}

if ! git log --format='%ae %ce' | grep -qF "$OLD"; then
    echo "The address given is not present in this history. Nothing to do."
    exit 0
fi

echo "Rewriting $(git rev-list --count HEAD) commits."
echo "Before:"
git log --format='  %h %ae' | sed -n '1,10p'

# A ref you can return to if anything looks wrong.
git branch -f pre-rewrite-backup HEAD
echo "backup branch 'pre-rewrite-backup' at $(git rev-parse --short HEAD)"

export FB_OLD="$OLD" FB_NEW="$NEW" FB_NAME="$NAME"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter '
if [ "$GIT_AUTHOR_EMAIL" = "$FB_OLD" ]; then
    export GIT_AUTHOR_EMAIL="$FB_NEW"
    [ -n "$FB_NAME" ] && export GIT_AUTHOR_NAME="$FB_NAME"
fi
if [ "$GIT_COMMITTER_EMAIL" = "$FB_OLD" ]; then
    export GIT_COMMITTER_EMAIL="$FB_NEW"
    [ -n "$FB_NAME" ] && export GIT_COMMITTER_NAME="$FB_NAME"
fi
' --tag-name-filter cat -- --all

echo
echo "After:"
git log --format='  %h %ae' | sed -n '1,10p'
echo
echo "Distinct addresses now: $(git log --format='%ae%n%ce' | sort -u | tr '\n' ' ')"

if git log --format='%ae %ce' | grep -qF "$OLD"; then
    echo "FAILED: the old address is still present. Not suggesting a push." >&2
    exit 1
fi

cat <<'MSG'

History is clean. To publish the rewrite:

    git push --force-with-lease origin main

Then add the new address at your forge's email settings so the commits link to
your account, and consider enabling "keep my email addresses private".

Once satisfied:  git branch -D pre-rewrite-backup
MSG
