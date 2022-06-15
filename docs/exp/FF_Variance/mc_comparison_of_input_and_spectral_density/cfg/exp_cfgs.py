from trieste.objectives.single_objectives import (
    GMM,
    HARTMANN_6_SEARCH_SPACE,
    SHEKEL_4_SEARCH_SPACE,
    Hartmann_3,
    SinLinear,
    hartmann_6,
    shekel_4,
)
from trieste.space import Box

SinLinear = {
    "Name": "SinLinear",
    "benchmark": SinLinear().objective(),
    "search_space": Box(*SinLinear.bounds),
}


GMM = {"Name": "GMM", "benchmark": GMM().objective(), "search_space": Box(*GMM.bounds)}


Hartmann3 = {
    "Name": "Hartmann3",
    "benchmark": Hartmann_3().objective(),
    "search_space": Box(*Hartmann_3.bounds),
}

Shekel4 = {
    "Name": "Shekel4",
    "benchmark": shekel_4,
    "search_space": SHEKEL_4_SEARCH_SPACE,
}


Hartmann6 = {
    "Name": "Hartmann6",
    "benchmark": hartmann_6,
    "search_space": HARTMANN_6_SEARCH_SPACE,
}
