#!/usr/bin/env bash
# 并行分片下载（绕单连接限速）
set -u
URL="$1"; OUT="$2"; N="${3:-16}"; SIZE="${4:-}"
cd "$(dirname "$OUT")" || exit 1
TMP="$(basename "$OUT").parts"; rm -rf "$TMP"; mkdir -p "$TMP"
if [ -z "$SIZE" ]; then
  SIZE=$(curl -sIL --max-time 20 "$URL" | awk '/[Cc]ontent-[Ll]ength/{l=$2} END{gsub(/\r/,"",l); print l}')
fi
echo "size=$SIZE parts=$N"
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  curl -s --max-time 600 -r "$s-$e" -o "$TMP/p$i" "$URL" &
done
wait
: > "$OUT"
for i in $(seq 0 $((N-1))); do cat "$TMP/p$i" >> "$OUT"; done
rm -rf "$TMP"
ls -l "$OUT" | awk '{print "final bytes:",$5}'
