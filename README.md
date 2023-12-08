# n_body_approx
## Files
[one_particle_withmlp.ipynb](one_particle_withmlp.ipynb) - training, results including models and data are logged into tensorboard \
[encoder_gravity_analysis.ipynb](encoder_gravity_analysis.ipynb) - analysis of tensorboard run id\
[gravity_simulator.ipynb](gravity_simulator.ipynb) - simulates and plots particles\
**models.py** - models...\
**loggers.py** - unified wrapper around various loggers like tensorboard, wandb,...\
**utils.py** - big box of various tools and stuff

## Tensorboard
tensorboard --logdir=runs_g_1p_encoder_mlp --port=8010\
http://localhost:8010/

## Weights & Biases
https://wandb.ai/martin-ka/projects

https://github.com/TheMightiestCarrot/n_body_approx/assets/68813317/fe1145ba-e814-42fe-bf28-716ca871909d

