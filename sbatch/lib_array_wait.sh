#!/bin/bash
# Helper sourced by the array-parent sbatch scripts (infer/morph/blender).
#
# A parent->array sbatch normally submits its array and exits immediately.
# That breaks `--dependency=afterok:<parent>` chaining: the dependent stage
# would start the moment the parent exits, i.e. before the array has run.
#
# wait_for_array <array_job_id> blocks until every task of the array has
# left the queue, then returns 0 iff all tasks reached COMPLETED. A parent
# that ends with `wait_for_array "$JID"; exit $?` therefore stays alive for
# the whole array and propagates its success/failure — so a downstream
# `afterok` dependent fires only after the array genuinely succeeded.
wait_for_array() {
    local jid="$1"
    if [ -z "$jid" ]; then
        echo "wait_for_array: no job id given"
        return 1
    fi

    # Brief grace period so the freshly-submitted array is visible to squeue.
    sleep 10
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
    # Let the accounting DB settle on final states.
    sleep 5

    # Count array tasks (JobID like 12345_3) that did not COMPLETE.
    local bad
    bad=$(sacct -j "$jid" -n -P -o JobID,State 2>/dev/null \
          | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2 !~ /^COMPLETED/ {c++} \
                       END {print c+0}')
    if [ "$bad" -gt 0 ]; then
        echo "ERROR: $bad task(s) of array $jid did not complete"
        return 1
    fi
    echo "All tasks of array $jid completed"
    return 0
}
