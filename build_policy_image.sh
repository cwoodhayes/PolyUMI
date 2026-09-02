#!/usr/bin/env bash
# build_policy_image.sh - Select a policy fork and build its image, in two stages.
#
# Sourced by train_policy.sh and serve_policy.sh so both roles run a byte-identical image --
# checkpoints are dill-pickled and must unpickle against the exact dependency tree they were
# trained with, so a training image and a serving image that differ is a class of bug that only
# shows up at load time.
#
# Also runnable on its own, to build without starting a run:
#   POLICY=vista ./build_policy_image.sh
#
# Two stages because a fork's Dockerfile builds with the FORK DIRECTORY as its context and so
# cannot see inference_server/. See docker/polyumi_inference.Dockerfile for why the alternatives
# (staging a copy into the fork; moving the fork to a repo-root context) are worse.
#
#   stage 1  the fork: conda env + torch + the policy code                ~23 min cold, cached after
#   stage 2  polyumi_inference on top                                     seconds
#
# Layer caching makes this cheaper than a single build would be: editing the shared library
# rebuilds only stage 2. The stage-1 solve is NOT shared between forks -- each has its own conda
# env, so the first build of a new policy pays the ~23 minutes again.

# Resolve POLICY -> POLICY_DIR, IMAGE, CONFIG_NAME, TRAIN_CMD, SERVE_CMD, DATASET_MNT.
#
# The wiring lives in the PARENT repo (config/policy.<name>.env), not in the forks: it is a set of
# facts about how PolyUMI drives a fork -- its path under external/, its image tag, its entrypoint
# -- which a submodule cannot know about itself and which must be readable before there is anything
# checked out to read. Same pattern as config/env.<hostname>.sh.
#
# Hyperparameters stay in each fork's own Hydra tree; CONFIG_NAME only selects which workspace yaml.
#
# Returns non-zero rather than calling exit, since this file is meant to be sourced: an exit here
# would kill an interactive shell that sourced it. Callers run under `set -e`, so an unchecked
# call still aborts the script.
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
    # config still works without editing config/.
    local image_override="${IMAGE:-}"
    local config_override="${CONFIG_NAME:-}"
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

build_policy_image() {
    local repo_root="$1"
    local image="$2"
    local policy_dir="$3"
    local base_image="${image}-base"

    echo ">> building ${base_image} from ${repo_root}/${policy_dir}"
    docker build -t "${base_image}" "${repo_root}/${policy_dir}"

    echo ">> layering polyumi_inference into ${image}"
    docker build \
        -t "${image}" \
        --build-arg "BASE=${base_image}" \
        -f "${repo_root}/docker/polyumi_inference.Dockerfile" \
        "${repo_root}/inference_server"
}

# Run standalone (build only) when executed rather than sourced.
if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    set -euo pipefail
    _REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    policy_select "${_REPO_ROOT}" "${POLICY:-dp}"
    build_policy_image "${_REPO_ROOT}" "${IMAGE}" "${POLICY_DIR}"
fi
