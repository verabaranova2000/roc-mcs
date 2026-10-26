"""Реестр приложений: метаданные отделены от способа запуска."""
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AppSpec:
    key: str
    title: str
    icon: str
    description: str
    available: bool = True
    single_instance: bool = True

APP_SPECS = (
    AppSpec("live_scans", "Live Scans", "app_live_scans.svg", "Потоковая РОК-карта и анализ серии", True, True),
    AppSpec("roi_explorer", "ROI Explorer", "app_roi_explorer.svg", "Исследование ROI в HDF5-сканах 2D-детектора", True, False),
    AppSpec("detector", "Детектор", "app_detector.svg", "Просмотр сигнала детектора — запланировано", False, True),
    AppSpec("archive", "Архив экспериментов", "app_archive.svg", "Поиск и просмотр архивных экспериментов — запланировано", False, True),
)
APP_BY_KEY = {spec.key: spec for spec in APP_SPECS}
