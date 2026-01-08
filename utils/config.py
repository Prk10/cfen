import yaml
import copy
import argparse

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def update_config(cfg, overrides):
    # overrides is a list like ["A.B val", "X.Y true"]
    new_cfg = copy.deepcopy(cfg)
    it = iter(overrides)
    for key, val in zip(it, it):
        keys = key.split(".")
        d = new_cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # parse bool/int/float
        sval = val
        if sval.lower() in ["true", "false"]:
            sval = sval.lower() == "true"
        else:
            try:
                if "." in sval:
                    sval = float(sval)
                else:
                    sval = int(sval)
            except:
                pass
        d[keys[-1]] = sval
    return new_cfg

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="KEY VALUE pairs to override config")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()
    return args
