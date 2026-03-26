import importlib
import sys
import types


def make_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def install_module(monkeypatch, name: str, module=None, **attrs):
    module = module or make_module(name, **attrs)
    monkeypatch.setitem(sys.modules, name, module)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = make_module(parent_name)
            monkeypatch.setitem(sys.modules, parent_name, parent)
        setattr(parent, child_name, module)

    return module


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
