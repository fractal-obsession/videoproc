#!/bin/bash

status_cmd="python /usr/src/app/videoproc.py --rootdir /opt/ComfyUI/ status"

python /opt/ComfyUI/main.py $COMFY_ARGS --output-directory /output --input-directory /input &

while :
do
  echo "waiting for ComfyUI to become available"
  sleep 3
  $status_cmd 1>/dev/null 2>/dev/null
  if [[ $? -eq 0 ]]
  then
    echo "ComfyUI server up"
    break
  fi
done

python /usr/src/app/videoproc.py --rootdir /opt/ComfyUI/ $@

while :
do
  sleep 10
  status=$($status_cmd)
  if [[ $status =~ "0 total" ]]
  then
    echo "done"
    exit 0
  fi
done
