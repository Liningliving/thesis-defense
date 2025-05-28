# SPDX-License-Identifier: AGPL-3.0-only

#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256  #,expandable_segments:True
echo $PYTORCH_CUDA_ALLOC_CONF