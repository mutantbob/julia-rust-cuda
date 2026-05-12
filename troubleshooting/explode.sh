#!/bin/sh

# for gentoo
export CUDA_TOOLKIT_PATH=/opt/cuda
export CUDA_OXIDE_LLC=/usr/lib/llvm/21/bin/llc


REPO=https://github.com/NVlabs/cuda-oxide.git
#REPO=$HOME/vendor/cuda-oxide

mkdir demo

cd demo

git clone $REPO
(cd cuda-oxide && git checkout c91cf17f9c5962ffb844cdbedf1163c716d12cd2)

echo "installing cargo-oxide from source"

cargo install --path cuda-oxide/crates/cargo-oxide cargo-oxide

rm -r a_b
cargo oxide new a_b

(cd a_b &&
     cargo oxide run)
