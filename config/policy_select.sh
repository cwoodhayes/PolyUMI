#!/usr/bin/env bash
# config/policy_select.sh - Resolve POLICY into the facts about how PolyUMI drives that fork.
#
# Sourced by train_policy.sh, serve_policy.sh and build_policy_image.sh. Reads
# config/policy.<name>.env and sets POLICY_DIR, IMAGE and TRAIN_CMD, plus SERVE_CMD and
# CONFIG_NAME where the fork has them. Why the wiring lives in this repo rather than in the
# submodule, and how to add a fork: docs/training-instructions.md § "Choosing a policy".

policy_select() {
    local repo_root="$1"
    local policy="$2"
    local env_file="${repo_root}/config/policy.${policy}.env"

    if [ ! -f "${env_file}" ]; then
        echo "error: unknown POLICY '${policy}' (no ${env_file#"${repo_root}/"})" >&2
        echo "available:" >&2
        for f in "${repo_root}"/config/policy.*.env; do
            f="${f##*/policy.}"
            echo "    ${f%.env}" >&2
        done
        return 1
    fi

    # A value already set in the calling shell wins over the file's, so a one-off tag or workspace
    # config still works without editing config/. The two are guarded differently because
    # CONFIG_NAME is optional: a fork's file may set none, which makes the `${override:-${VAR}}`
    # form an unbound-variable error under `set -u`.
    local image_override="${IMAGE:-}"
    local config_override="${CONFIG_NAME:-}"
    # A fork with no inference server sets no SERVE_CMD. Cleared rather than left alone so a value
    # inherited from the caller's environment cannot stand in for one the fork does not have.
    # shellcheck disable=SC2034  # read by serve_policy.sh, which sources this
    SERVE_CMD=""
    # shellcheck disable=SC1090
    source "${env_file}"
    IMAGE="${image_override:-${IMAGE}}"
    if [ -n "${config_override}" ]; then
        CONFIG_NAME="${config_override}"
    fi

    # An uninitialised submodule is an existing but EMPTY directory, so check for what stage 1
    # actually needs rather than for the directory.
    # shellcheck disable=SC2153  # POLICY_DIR comes from the sourced env file above
    if [ ! -f "${repo_root}/${POLICY_DIR}/Dockerfile" ]; then
        echo "error: no ${POLICY_DIR}/Dockerfile -- run 'git submodule update --init ${POLICY_DIR}'" >&2
        return 1
    fi
}
