#!/usr/bin/env bash
# Run compose services one after the other on the DGX Spark, on the host. Each service's container is
# edge-grader-<service>. Stops at the first that doesn't exit 0. AFTER=<container> waits for it first.
#
#   AFTER=edge-grader-bench-asap2-think nohup scripts/spark_queue.sh reasons-beetle-saf bench-scientsbank \
#     > data/logs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
if [ -n "${AFTER:-}" ]; then
  echo "$(date -u +%FT%TZ) waiting for $AFTER"
  code=$(docker wait "$AFTER")
  echo "$(date -u +%FT%TZ) $AFTER exited $code"
  [ "$code" = 0 ] || exit "$code"
fi
for service in "$@"; do
  echo "$(date -u +%FT%TZ) start $service"
  docker compose up -d --force-recreate "$service" || exit 1
  code=$(docker wait "edge-grader-$service")
  echo "$(date -u +%FT%TZ) $service exited $code"
  [ "$code" = 0 ] || exit "$code"
done
echo "$(date -u +%FT%TZ) queue done"
