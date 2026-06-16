import yaml

class Config:
    """Recursively wraps a dict so fields are accessible as attributes."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, str):
                value = value.format(**data)
                data[key] = value
                print(key, data[key])
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({fields})"


def load_config(path="config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

if __name__ == '__main__':
    cfg = load_config('mock_config_v2.1.yaml')
    CONFIG = cfg.CONFIG 
    print(CONFIG.mock_bgs_clus_data.format(phase=1, real=2))
    print(CONFIG.mock_bgs_clus_rand)