#!/usr/bin/env sh
set -e

TOOLS=$CAFFE/build/tools

$TOOLS/caffe train \
  --solver=models/cifar10_quick/solver.prototxt $@
