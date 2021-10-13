#!/bin/bash

# Generate synthetic data
python generate_synthetic_simplex_data.py

# Run the algorithm
python demo_post-nonlinear_simplex_synthetic.py \
--s_dim=3 \
--batch_size=1000 \
--num_epochs=20 \
--inner_iters=100 \
--learning_rate=1e-3 \
--rho=1e2 \
--model_file_name="best_model_simplex.pth" \
--f_num_layers=3 \
--f_hidden_size=128 \
--q_num_layers=3 \
--q_hidden_size=128 \

