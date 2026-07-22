import yaml
import numpy as np
import os

class Config:
    """Recursively wraps a dict so fields are accessible as attributes."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, str):
                value = value.format(**data)
                data[key] = value
                #print(key, data[key])
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({fields})"


def load_config(path="config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

def get_files_to_download(cfg):

    files = []
    for v in cfg.__dict__.values(): 
        if(type(v)!=str):
            continue
        if v.endswith('.fits') or v.endswith('.csv') or v.endswith('.hdf5'):
            if 'phase' in v:
                for phase in range(25):
                    if 'real' in v:
                        for real in range(27):
                            files.append(v.format(phase=phase, real=real))
                    else:
                        files.append(v.format(phase=phase))
            else:
                files.append(v)
    return np.array(files)


if __name__ == '__main__':
    cfg = load_config('config_files/mock_config_v3.0.yaml')
    files = get_files_to_download(cfg)
    exist = np.array([os.path.exists(f) for f in files])
    print(files.size, np.unique(files).size, np.sum(exist))