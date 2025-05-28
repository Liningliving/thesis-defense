# SPDX-License-Identifier: AGPL-3.0-only

#!/bin/bash
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# python Run/run.py
printenv TRANSFORMERS_OFFLINE
printenv HF_HUB_OFFLINE
printenv HF_DATASETS_OFFLINE