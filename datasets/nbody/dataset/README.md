#### Creating N-Body data
To recreate the datasets used in this work, navigate to ```nbody/dataset/``` and run either
```bash
python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
```
or
```bash
python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small