#!/bin/bash
#SBATCH --job-name=color_correction
#SBATCH -A dash_agir
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o color_correction-%j.out



