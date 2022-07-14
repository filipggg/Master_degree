#!/bin/bash -xe

challenge_dir="$1"

cp train.sh predict.sh prepare.sh ${challenge_dir}/

cd $challenge_dir

mkdir code

cd code

git submodule add https://github.com/scardine/image_size
git submodule add https://github.com/pytorch/vision
git submodule add https://github.com/filipggg/Master_degree.git
