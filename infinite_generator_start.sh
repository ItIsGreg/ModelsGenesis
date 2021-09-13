#!/bin/bash
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/sda1/luna16 \
--save generated_cubes
done
