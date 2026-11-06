from weight_space_programming_skill import WeightSpaceProgram


def test_no_adapters(monkeypatch):
    # Temporarily clear adapters by monkeypatching the import
    monkeypatch.setattr(
        'weight_space_programming_skill.src.core.importlib.import_module',
        lambda name: type('mod', (), {'__path__': []})
    )
    wsp = WeightSpaceProgram()
    result = wsp.run('test prompt')
    assert result.startswith('[No adapters]')


def test_generic_adapter(monkeypatch):
    # Ensure generic_adapter is loaded
    wsp = WeightSpaceProgram()
    result = wsp.run('hello')
    # Expect at least generic output present
    assert '[generic] hello' in result
