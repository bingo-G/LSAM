"""
Simple class/function registry utility.
"""


class Registry:
    """A registry that maps names to classes/functions."""

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name: str = None):
        """Decorator to register a class/function."""
        def wrapper(cls_or_fn):
            key = name or cls_or_fn.__name__
            if key in self._registry:
                raise KeyError(f'{key} already registered in {self._name}')
            self._registry[key] = cls_or_fn
            return cls_or_fn
        return wrapper

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f'{name} not found in {self._name}. Available: {list(self._registry.keys())}')
        return self._registry[name]

    def keys(self):
        return self._registry.keys()

    def __contains__(self, name: str):
        return name in self._registry

    def __repr__(self):
        return f'Registry({self._name}, keys={list(self._registry.keys())})'


DATASETS = Registry('datasets')
MODELS = Registry('models')
LOSSES = Registry('losses')
