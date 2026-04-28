#!/usr/bin/env bash

# https://stackoverflow.com/questions/79297468/how-to-write-a-gpu-worker-pool-to-run-multiple-tasks-at-the-same-time-in-bash

mkfifo free_gpus
exec 3<>free_gpus

# write the GPUs you want to use (0,1,2,3) into the fifo.
printf '%s\n' 0 1 2 3 4 5 6 7 >&3  &


allscripts=(

)


for exe in ${allscripts[@]}
do
    read gpu
    sleep 3
    # sleep 15
    { printf "GPU ${gpu}  for ${exe}\n" >> "allscripts_logs.txt"; CUDA_VISIBLE_DEVICES="$gpu" "$exe"; echo "$gpu" >&3; } &
done <&3
wait

# rm free_gpus


