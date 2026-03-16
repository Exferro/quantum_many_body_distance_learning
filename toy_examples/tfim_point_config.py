from nestconf import Configurable


class PhaseDiagramPoint(Configurable):
    system_size: int = None
    transverse_field: float = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
