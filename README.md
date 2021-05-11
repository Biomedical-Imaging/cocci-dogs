# Cocci Dogs

### Hyperparameter Search
```bash
python runner.py --gpus 0,1,2,3 --max_concurrent 4
```

### Cross Validation
```bash
python main.py --sherpa_trial <sherpa-trial-number> --notsherpa --gpu 0
```

### Confusion Matrix
```bash
python main.py --sherpa_trial <sherpa-trial-number> --notsherpa --gpu 0 --cm
```

### Class Activation Maps
```bash
python main.py --sherpa_trial <sherpa-trial-number> --notsherpa --gpu 0 --cam --model_path Models/<sherpa-trial-number>/00001.h5
```