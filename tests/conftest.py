import importlib
import sys
import types

import pytest


def _make_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def module_factory():
    return _make_module


@pytest.fixture
def fresh_import(monkeypatch):
    def _fresh_import(module_name: str, stub_modules=None, clear_modules=None):
        for name in clear_modules or []:
            sys.modules.pop(name, None)
        sys.modules.pop(module_name, None)

        for name, module in (stub_modules or {}).items():
            monkeypatch.setitem(sys.modules, name, module)

        return importlib.import_module(module_name)

    return _fresh_import
