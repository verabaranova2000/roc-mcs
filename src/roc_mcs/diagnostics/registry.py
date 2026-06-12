from roc_mcs.plots import plot_phase_diagnostics

DIAGNOSTICS = {
    "phase": plot_phase_diagnostics,
}


def parse_diagnostics(value):
    if not value:
        return []

    if isinstance(value, str):
        names = [name.strip() for name in value.split(",") if name.strip()]
    elif isinstance(value, list):
        names = value
    else:
        raise TypeError("diagnostics must be a string or a list")

    return [DIAGNOSTICS[name] for name in names]