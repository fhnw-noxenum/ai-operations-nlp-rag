#!/bin/sh
set -eu

LANGUAGE="${1:-en}"
REPO_URL="https://github.com/kubernetes/website.git"
WORKDIR="${KUBERNETES_DOCS_WORKDIR:-/tmp/kubernetes-docs}"
SOURCE_DIR="$WORKDIR/content/$LANGUAGE/docs"
TARGET_DIR="data/kubernetes/$LANGUAGE"

rm -rf "$WORKDIR" "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$WORKDIR"
git -C "$WORKDIR" sparse-checkout set "content/$LANGUAGE/docs"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "No Kubernetes docs found for language '$LANGUAGE'."
    echo "Expected path: content/$LANGUAGE/docs"
    exit 1
fi

find "$SOURCE_DIR" -type f \( -name "*.md" -o -name "*.txt" \) | while read -r file; do
    relative_path="${file#$SOURCE_DIR/}"
    mkdir -p "$TARGET_DIR/$(dirname "$relative_path")"
    cp "$file" "$TARGET_DIR/$relative_path"
done

rm -rf "$WORKDIR"

echo "Downloaded Kubernetes $LANGUAGE docs to $TARGET_DIR"
