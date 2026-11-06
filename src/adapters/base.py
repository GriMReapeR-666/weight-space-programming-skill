class BaseAdapter:
    """Abstract base class for all adapters.
    Subclasses must implement the ``run`` method which takes a prompt string
    and returns a processed result string.
    """

    def run(self, prompt: str) -> str:
        raise NotImplementedError("Adapter must implement the run method")
