#!/bin/bash

# setup_exiftool.sh

set -e

EXIFTOOL_VERSION="13.42"
INSTALL_DIR="$HOME/bin"
TOOL_DIR="$INSTALL_DIR/Image-ExifTool-$EXIFTOOL_VERSION"
URL="https://sourceforge.net/projects/exiftool/files/Image-ExifTool-$EXIFTOOL_VERSION.tar.gz/download"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download and extract ExifTool
curl -L -o "exiftool.tar.gz" "$URL"
tar -xzf "exiftool.tar.gz"

# Create symlink to `exiftool` in $HOME/bin
ln -sf "$TOOL_DIR/exiftool" "$INSTALL_DIR/exiftool"

echo "✅ ExifTool installed at: $INSTALL_DIR/exiftool"
echo "⚠️ Add the following line to your shell config if not already done:"
echo "export PATH=\"\$HOME/bin:\$PATH\""
echo "By running: echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bashrc"
echo "Then, reload your shell or run: source ~/.bashrc"
