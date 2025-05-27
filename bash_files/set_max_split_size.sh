#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256  #,expandable_segments:True
echo $PYTORCH_CUDA_ALLOC_CONF