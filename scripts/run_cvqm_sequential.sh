#!/usr/bin/env bash
# =============================================================================
#  run_cvqm_sequential.sh — 复现 LSAM 在 CVQM 全量数据上的性能
#  （单个序列依次推理：batch_size=1，逐个 (ref, dis) 顺序前向）
# -----------------------------------------------------------------------------
#  【当前待运行命令】(可直接复制到终端；换机后先重新生成 resolved manifest)
#
#    # 1) 一次性：把 cvqm_manifest.csv 解析成绝对路径 (Phase1 在 /data2, Phase2 在 /data3)
#    cd /data2/guanfb/Ali/hmf_vqa/LSAM_infer_crosscheck && \
#    python3 scripts/check_cvqm_data.py \
#        --cvqm_root /data2/datasets/CVQM /data3/datasets/VQA/CVQM/decoded \
#        --ref_root  /data2/datasets/CVQM/all_original_convert_10bit_ffmpeg_correct_10s \
#        --layout auto --manifest labels/cvqm_manifest.csv \
#        --output /data2/guanfb/Ali/hmf_vqa/LSAM/outputs/cvqm_resolved.csv
#
#    # 2) 单序列依次推理（走完 851 条 CVQM 数据，输出预测 + SRCC/PLCC/KRCC/RMSE）
#    cd /data2/guanfb/Ali/hmf_vqa/LSAM && \
#    GPU=0 bash scripts/run_cvqm_sequential.sh
#
#    # 后台运行并记录日志：
#    cd /data2/guanfb/Ali/hmf_vqa/LSAM && \
#    GPU=0 nohup bash scripts/run_cvqm_sequential.sh > outputs/cvqm_sequential.log 2>&1 &
# -----------------------------------------------------------------------------
#  环境变量覆盖 (可选):
#      GPU=0            使用的 GPU 序号 (默认 0)
#      CKPT=...         权重路径 (默认 weights/lsam.pth)
#      MANIFEST=...     已解析 manifest (默认 outputs/cvqm_resolved.csv)
#      OUTPUT=...       预测结果 CSV (默认 outputs/cvqm_scores.csv)
#      NUM_WORKERS=4    DataLoader IO worker 数 (默认 4；不影响“单序列依次”语义)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-weights/lsam.pth}"
MANIFEST="${MANIFEST:-outputs/cvqm_resolved.csv}"
OUTPUT="${OUTPUT:-outputs/cvqm_scores.csv}"

GPU="${GPU:-0}"
# 单个序列依次推理：batch_size 固定为 1（每次前向只处理一个视频）
BATCH_SIZE=1
NUM_WORKERS="${NUM_WORKERS:-4}"
# 预处理路径：GPU (mode 2) 与 CPU 预处理数值一致，可在 GPU 加速下复现基准性能。
# 如需强制走 CPU 预处理，设 GPU_PREPROCESS=0。
GPU_PREPROCESS="${GPU_PREPROCESS:-2}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    echo "  请先运行本文件头部【当前待运行命令】第 1 步生成 resolved manifest。" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "=============================================================="
echo "  LSAM CVQM 单序列依次推理"
echo "    Checkpoint : ${CKPT}"
echo "    Manifest   : ${MANIFEST}  ($(( $(wc -l < "${MANIFEST}") - 1 )) 条)"
echo "    Output     : ${OUTPUT}"
echo "    GPU        : ${GPU}"
echo "    Batch size : ${BATCH_SIZE}  (单序列依次)"
echo "    Workers    : ${NUM_WORKERS}"
echo "    Preprocess : $( [[ "${GPU_PREPROCESS}" == "0" ]] && echo 'CPU' || echo "GPU mode=${GPU_PREPROCESS} (已对齐CPU)" )"
echo "=============================================================="

HMF_VQA_GPU_PREPROCESS="${GPU_PREPROCESS}" CUDA_VISIBLE_DEVICES="${GPU}" python -B -u infer.py \
    --eval_ckpt   "${CKPT}" \
    --manifest    "${MANIFEST}" \
    --output      "${OUTPUT}" \
    --batch_size  "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
