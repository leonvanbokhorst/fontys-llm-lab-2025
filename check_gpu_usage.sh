#!/bin/bash
usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [ "$usage" -gt 95 ]; then
  echo "$(date): GPU usage high at ${usage}%!" >> /app/gpu_alert.log
fi 