# n_body_approx
## Files
[one_particle_withmlp.ipynb](one_particle_withmlp.ipynb) - training, results including models and data are logged into tensorboard \
[encoder_gravity_analysis.ipynb](encoder_gravity_analysis.ipynb) - analysis of tensorboard run id\
[gravity_simulator.ipynb](gravity_simulator.ipynb) - simulates and plots particles\
**models.py** - models...\
**utils.py** - big box of various tools and stuff

## Tensorboard
tensorboard --logdir=runs_g_1p_encoder_mlp --port=8010\
http://localhost:8010/