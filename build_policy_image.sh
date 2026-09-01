#!/usr/bin/env bash
# build_policy_image.sh - Build a policy fork's image, in two stages.
#
# Sourced by train_policy.sh and serve_policy.sh so both roles run a byte-identical image --
# checkpoints are dill-pickled and must unpickle against the exact dependency tree they were
# trained with, so a training image and a serving image that differ is a class of bug that only
# shows up at load time.
#
# Which fork gets built is config/policy_select.sh's job. Also runnable on its own, to build
# without starting a run:
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
    # shellcheck source=config/policy_select.sh
    source "${_REPO_ROOT}/config/policy_select.sh"
    policy_select "${_REPO_ROOT}" "${POLICY:-dp}"
    build_policy_image "${_REPO_ROOT}" "${IMAGE}" "${POLICY_DIR}"
fi
