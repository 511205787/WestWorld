#!/usr/bin/env bash
for i in 1 3 4 5; do
  echo "Clearing GPU $i …"
  for fd in /proc/*/fd/*; do
    link=$(readlink "$fd" 2>/dev/null) || continue
    if [[ "$link" == "/dev/nvidia$i" ]]; then
      pid=${fd#"/proc/"}; pid=${pid%%"/fd/"*}
      # only kill your own process
      if ps -o user= -p "$pid" | grep -q "^$USER\$"; then
        echo "  Killing -9 $pid"
        kill -9 "$pid"
      fi
    fi
  done
done
