import importlib

for name in ['trl', 'peft', 'transformers', 'torch', 'bitsandbytes', 'datasets', 'accelerate']:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, '__version__', '?'))
    except Exception as exc:
        print(name, 'MISSING', type(exc).__name__, exc)
try:
    import trl
    from trl import DPOTrainer, DPOConfig
    print('DPOTrainer OK')
except Exception as exc:
    print('DPOTrainer unavailable', exc)
import torch
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
