#!/bin/bash
#SBATCH --job-name="vscode-tunnel"
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o vscode-tunnel-%j.out

# Set VS Code data directories to project space (replace with your project name)
export VSCODE_CLI_DATA_DIR="/project/dash_agir/matthew.kutugata/vscode-data/cli"
export VSCODE_AGENT_FOLDER="/project/dash_agir/matthew.kutugata/vscode-data/server"

# Create directories if they don't exist
mkdir -p $VSCODE_CLI_DATA_DIR
mkdir -p $VSCODE_AGENT_FOLDER

# Set up VS Code CLI location
CODE_CLI="$HOME/code"

# Run the tunnel
$CODE_CLI tunnel --accept-server-license-terms --name crs-mk-$(date +%Y%m%d)
