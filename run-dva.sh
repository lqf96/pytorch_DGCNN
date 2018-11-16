#! /bin/sh

# Available GPUs
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# General settings
# Model name
gm=DGCNN
# Device ("gpu" or "cpu")
device=gpu
# Convolution size
conv_size="32-32-32-1"
# Sort pooling k
# (If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer)
sortpooling_k=0.6
# Final dense layer's input dimension
# (Automatically determined)
fp_len=0
# Final dense layer's hidden size
n_hidden=128
# Batch size
batch_size=50
# Dropout layer
dropout=True
# Weight decay
weight_decay=0.01

# Learning rate
learning_rate=0.000001
# Number of epochs
n_epochs=100

# Run training program
time python main.py \
    -data DVA \
    -test_number ${N_TESTS} \
    -training-set train.txt \
    -testing-set test.txt \
    -learning_rate ${learning_rate} \
    -num_epochs ${n_epochs} \
    -hidden ${n_hidden} \
    -latent_dim ${conv_size} \
    -sortpooling_k ${sortpooling_k} \
    -out_dim ${fp_len} \
    -batch_size ${batch_size} \
    -gm ${gm} \
    -mode ${device} \
    -dropout ${dropout} \
    -weight_decay ${weight_decay} \
    -printAUC true
