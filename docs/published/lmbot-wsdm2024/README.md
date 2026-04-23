# LMbot: Language Model-Enhanced Social Bot Detection

**Published:** WSDM 2024  
**Status:** Immutable published work

## Citation

```bibtex
@inproceedings{lmbot2024,
  title={LMbot: Language Model-Enhanced Social Bot Detection},
  author={[Authors]},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  year={2024}
}
```

## Overview

LMbot combines language models and graph neural networks for social bot detection on Twitter/X platform. This directory contains the exact code used for the WSDM 2024 paper.

## Reproduction

### Requirements

```bash
pip install -r ../../../requirements.txt
```

### Running LMbot

**Full pipeline:**
```bash
python src/main.py --dataset TwiBot-20 --seeds 1,2,3,4,5
```

**GNN-only baseline:**
```bash
python src/main.py --dataset TwiBot-20 --mode gnn_only
```

**LM-only baseline:**
```bash
python src/main.py --dataset TwiBot-20 --mode lm_only
```

### Expected Results

On TwiBot-20 dataset:
- **LMbot GNN**: Accuracy 0.8533, F1 0.8732
- **LMbot LM**: Accuracy 0.8554, F1 0.8756

## Files

- `src/main.py` - Main entry point
- `src/trainer.py` - Training loop
- `src/LM.py` - Language model component
- `src/GNNs.py` - Graph neural network models
- `src/SimpleHGN.py` - Heterogeneous graph network
- `src/RGT.py` - Relational graph transformer
- `src/dataloader.py` - Data loading utilities
- `src/utils.py` - Utility functions
- `src/parser_args.py` - Argument parser
- `src/model_building.py` - Model construction
- `configs/lmbot.yaml` - Configuration file

## Notes

- This is the **original published implementation** - not maintained
- For active development, see `../../../src/lmbot/`
- Uses original data splits and preprocessing
- Results may vary slightly due to hardware/library versions

## Known Issues

None reported at publication time.

## Contact

For questions about this published work, refer to the paper or contact the authors.
