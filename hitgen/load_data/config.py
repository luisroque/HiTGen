from hitgen.load_data.m1 import M1Dataset
from hitgen.load_data.m4 import M4Dataset
from hitgen.load_data.m3 import M3Dataset
from hitgen.load_data.tourism import TourismDataset
from hitgen.load_data.m5 import M5Dataset
from hitgen.load_data.labour import LabourDataset
from hitgen.load_data.traffic import TrafficDataset
from hitgen.load_data.wiki2 import Wiki2Dataset
from hitgen.load_data.etth1 import ETTh1Dataset
from hitgen.load_data.etth2 import ETTh2Dataset
from hitgen.load_data.ettm1 import ETTm1Dataset
from hitgen.load_data.ettm2 import ETTm2Dataset
from hitgen.load_data.ecl import ECLDataset
from hitgen.load_data.trafficl import TrafficLDataset
from hitgen.load_data.weather import WeatherDataset


DATASETS = {
    "Tourism": TourismDataset,
    "M1": M1Dataset,
    "M3": M3Dataset,
    "M4": M4Dataset,
    "M5": M5Dataset,
    "Labour": LabourDataset,
    "Traffic": TrafficDataset,
    "Wiki2": Wiki2Dataset,
    "ETTh1": ETTh1Dataset,
    "ETTh2": ETTh2Dataset,
    "ETTm1": ETTm1Dataset,
    "ETTm2": ETTm2Dataset,
    "ECL": ECLDataset,
    "TrafficL": TrafficLDataset,
    "Weather": WeatherDataset,
}

DATASETS_FREQ = {
    "Tourism": ["Monthly", "Quarterly"],
    "M1": ["Monthly", "Quarterly"],
    "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
    "M5": ["Daily"],
    "Labour": ["Monthly"],
    "Traffic": ["Daily"],
    "Wiki2": ["Daily"],
    "ETTh1": ["Hourly"],
    "ETTh2": ["Hourly"],
    # "ETTm1": ["15T"],
    # "ETTm2": ["15T"],
    # "ECL": ["15T"],
    # "TrafficL": ["15T"],
    # "Weather": ["10M"],
}

REFERENCE_MODELS = [
    "ETS",
    "Theta",
    "SES",
    "Croston",
    "ARIMA",
    "RWD",
    "SNaive",
    "TiDE",
    "TCN",
    "DeepAR",
    "Informer",
    "NHITS",
    "RNN",
]

N = [1, 2, 3, 4, 5, 6]
SAMPLE_COUNT = 50
