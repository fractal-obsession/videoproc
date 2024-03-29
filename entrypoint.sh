#!/bin/bash

status_cmd="python3 /usr/src/app/videoproc.py --rootdir /opt/ComfyUI/ status"

python3 /opt/ComfyUI/main.py $COMFY_ARGS --output-directory /output --input-directory /input &

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

echo "$VIDEOPROC_ARGS"

eval /usr/src/app/videoproc.py --rootdir /opt/ComfyUI/ $@ $VIDEOPROC_ARGS

while :
do
  sleep 10
  status=$($status_cmd)
  echo "$status"
  if [[ $status == "0 running 0 pending 0 total" ]]
  then
    break
  fi
done

echo "done"
exit 0
