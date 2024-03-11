# n_body_approx
## Contents
[set_transformer_one_particle_withmlp_v2.ipynb](set_transformer_one_particle_withmlp_v2.ipynb) - training, results including models and data are logged into tensorboard \
[set_transformer_encoder_gravity_analysis.ipynb](set_transformer_encoder_gravity_analysis.ipynb) - analysis of tensorboard run id\
[gravity_simulator.ipynb](learning_materials%2Fgravity_simulator.ipynb) - simulates and plots particles\
**models.py** - models...\
**utils** - big box of various tools and stuff\
**utils\loggers.py** - unified wrapper around various loggers like tensorboard, wandb,...\

## Tensorboard
tensorboard --logdir=runs_g_1p_encoder_mlp --port=8010\
http://localhost:8010/

## Weights & Biases
https://wandb.ai/martin-ka/projects

https://github.com/TheMightiestCarrot/n_body_approx/assets/68813317/fe1145ba-e814-42fe-bf28-716ca871909d

