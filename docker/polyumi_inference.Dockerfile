# Layers polyumi_inference onto the diffusion-policy image.
#
# Two stages, rather than one, because the fork's own Dockerfile builds with the FORK DIRECTORY as
# its context and so cannot see inference_server/. The alternatives were worse: staging a copy of
# the library into the fork re-introduces exactly the duplicated file this library exists to delete,
# and switching the fork to a repo-root context would make a submodule's Dockerfile reference
# parent-repo paths, so the fork could no longer be built on its own.
#
# This build's context is inference_server/ itself -- 180 KB, six files -- so the extra stage costs
# a couple of seconds. It is also cheaper than a single build would be: editing the library rebuilds
# only these two layers and never touches the ~23-minute conda solve underneath.
#
# Built by train_policy.sh and serve_policy.sh, which both run both stages so the two roles keep
# running a byte-identical image (checkpoints are dill-pickled against the exact dep tree).
#
#   docker build -t polyumi-dp-base external/polyumi_diffusion_policy
#   docker build -t polyumi-dp -f docker/polyumi_inference.Dockerfile inference_server

ARG BASE=polyumi-dp-base
FROM ${BASE}

COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/polyumi_inference

# --no-deps: the image already has fastapi, uvicorn, pydantic and numpy, and re-resolving them here
# risks floating a pin the checkpoint was dill-pickled against. No [server] extra for the same
# reason -- the deps are present, they are just not pip's to choose.
#
# `micromamba run -n umi` explicitly, matching the base Dockerfile's own pip layers, rather than
# relying on MAMBA_DOCKERFILE_ACTIVATE: that is a build-time ARG and does not survive into a
# derived build.
RUN micromamba run -n umi pip install --no-cache-dir --no-deps /opt/polyumi_inference
