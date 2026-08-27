#!/usr/bin/env bash
# build_policy_image.sh - Build the PolyUMI diffusion-policy image, in two stages.
#
# Sourced by train_policy.sh and serve_policy.sh so both roles run a byte-identical image --
# checkpoints are dill-pickled and must unpickle against the exact dependency tree they were
# trained with, so a training image and a serving image that differ is a class of bug that only
# shows up at load time.
#
# Two stages because the fork's Dockerfile builds with the FORK DIRECTORY as its context and so
# cannot see inference_server/. See docker/polyumi_inference.Dockerfile for why the alternatives
# (staging a copy into the fork; moving the fork to a repo-root context) are worse.
#
#   stage 1  the fork: conda env + torch + diffusion_policy      ~23 min cold, cached after
#   stage 2  polyumi_inference on top                            seconds
#
# Layer caching makes this cheaper than a single build would be: editing the shared library
# rebuilds only stage 2.

build_policy_image() {
    local repo_root="$1"
    local image="$2"
    local base_image="${image}-base"

    echo ">> building ${base_image} from ${repo_root}/external/polyumi_diffusion_policy"
    docker build -t "${base_image}" "${repo_root}/external/polyumi_diffusion_policy"

    echo ">> layering polyumi_inference into ${image}"
    docker build \
        -t "${image}" \
        --build-arg "BASE=${base_image}" \
        -f "${repo_root}/docker/polyumi_inference.Dockerfile" \
        "${repo_root}/inference_server"
}
