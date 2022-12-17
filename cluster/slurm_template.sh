#!/bin/bash
# shellcheck disable=SC2206
${JOB_OPTS}

# Load modules or your own conda environment here
${LOAD_ENV}
export SLURM_JOB_NAME=bash

# ===== Call your code below =====
${COMMAND_PLACEHOLDER}