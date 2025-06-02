# SPDX-License-Identifier: AGPL-3.0-only
#!/bin/bash
# run_all.sh — 一键串联 1→4
# cd bash_files/

# —— Step 1: 解压RGB-D帧，并将RGB帧与Depth帧对齐 ——
echo "===== Step 1: 解压RGB-D帧，并将RGB帧与Depth帧对齐 ====="
bash bash_files/unzip_sequence_align_RGB.sh
if [ $? -ne 0 ]; then
  echo "Error: Step 1 关键帧提取失败，退出演示。" >&2
  exit 1
fi
echo "Step 1 完成，对齐后的RGB帧储存在os.path.join(CONF.PATH.R3SCAN_DATA_OUT,"aligned_RGB")下。"
echo ""

# —— Step 2：筛选关键帧，并将点云数据投影到关键帧上 ——
echo "===== Step 2: 筛选关键帧，并将点云数据投影到关键帧上 ====="
bash bash_files/run_get_object_fram.sh
if [ $? -ne 0 ]; then
  echo "Error: Step 2 数据预处理失败，退出演示。" >&2
  exit 1
fi
echo "Step 2 完成，投影后的数据储存在os.path.join(CONF.PATH.R3SCAN_DATA_OUT,"view")下。"
echo ""

# —— Step 3：大模型调用 (生成JSON三元组 ) ——
echo "===== Step 3: 调用大模型生成JSON三元组 ====="
bash bash_files/get_relationship.sh
if [ $? -ne 0 ]; then
  echo "Error: Step 3 LLM 调用失败，退出演示。" >&2
  exit 1
fi
echo "Step 3 完成,被标注物体间的关系在os.path.join(CONF.PATH.R3SCAN_DATA_OUT,"object")下。"
echo "被标注物体与其周围物体间及周围物体之间关系在os.path.join(CONF.PATH.R3SCAN_DATA_OUT,"relationship"）下。"

echo "===== 执行完毕 ====="