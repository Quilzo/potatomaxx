#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Remove an old email address from this repository's commit history.
#
# Git stores the author and committer address in every commit object, so the
# seven commits made before the identity was corrected still carry the old one,
# and GitHub shows it publicly. Fixing that means rewriting those commits, which
# changes their hashes and therefore requires a force-push.
#
# This is safe here and would not be in general: the repository has 0 forks,
# 0 stars and 0 watchers, so nobody holds a clone whose history this would
# invalidate. Check that again before running:
#
#   gh repo view Quilzo/potatomaxx --json forkCount,stargazerCount
#
# Run from the repository root. It is deliberately a script you execute yourself
# rather than something run on your behalf, because force-pushing over published
# history is not a decision to delegate.
set -eu

OLD='rashik.adhikari@dotcms.com'
NEW='rashik.cybersec@gmail.com'
NAME='rsh1k'

echo "Before:"
git log --format='  %h %ae' | sed -n '1,10p'

# Belt and braces: take a backup ref you can return to.
git branch -f pre-rewrite-backup HEAD
echo "backup branch 'pre-rewrite-backup' created at $(git rev-parse --short HEAD)"

FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter "
if [ \"\$GIT_AUTHOR_EMAIL\" = '$OLD' ]; then
    export GIT_AUTHOR_EMAIL='$NEW'
    export GIT_AUTHOR_NAME='$NAME'
fi
if [ \"\$GIT_COMMITTER_EMAIL\" = '$OLD' ]; then
    export GIT_COMMITTER_EMAIL='$NEW'
    export GIT_COMMITTER_NAME='$NAME'
fi
" --tag-name-filter cat -- --all

echo
echo "After:"
git log --format='  %h %ae' | sed -n '1,10p'
echo
echo "Distinct addresses now: $(git log --format='%ae%n%ce' | sort -u | tr '\n' ' ')"

if git log --format='%ae %ce' | grep -q "$OLD"; then
    echo "FAILED: the old address is still present. Not pushing." >&2
    exit 1
fi

echo
echo "History is clean. To publish the rewrite:"
echo
echo "    git push --force-with-lease origin main"
echo
echo "Then, so the commits link to your GitHub account, add $NEW at"
echo "https://github.com/settings/emails and consider enabling"
echo "'Keep my email addresses private'."
echo
echo "Once you are satisfied:  git branch -D pre-rewrite-backup"
