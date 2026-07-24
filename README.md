# Codec_TTS
This is the dev README.md of codec TTS model, which will be used to used to train the models using codec as well as the testing modules are implemented.

- [Architecture](#architecture)
- [Training the Model](#training-steps)
- [Testing the Model](#testing-the-model)
- [Parameters](#parameters)

### Architecture
### Training Steps
Activate the cirtual environment
```bash
source .venv/bin/activate
```

Install training dependencies
```bash
pip install -r requirements/train.txt
```

1. Record Hindi speech data (generates sentences with a local Ollama model, records your mic, saves `data/raw/hindi/NNN.txt` + `NNN.wav`)
```bash
python scripts/record_studio.py
# open http://localhost:7861
```

2. Tokenize the recorded dataset into Mimi codec codes
```bash
python scripts/tokenize_dataset.py --config configs/datasets/hindi.yaml
```

3. Train the backbone + depth transformer
```bash
python scripts/train.py --config configs/training/hindi_training.yaml
```

4. Monitor training progress (loss curves)
```bash
python scripts/monitor.py
# open http://localhost:7860
```

### Testing the Model
### Parameters