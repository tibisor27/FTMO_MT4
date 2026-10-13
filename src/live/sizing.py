"""
Politica de marime, citita din tabelul rezolvat prin programare dinamica.

De unde vine tabelul
--------------------
`scripts/final_policy.py` rezolva o problema HJB pe (faza, zile ramase, echitate) si
alege, pentru fiecare stare, tinta de volatilitate zilnica care maximizeaza valoarea
asteptata a programului. Nucleul de tranzitie nu e gaussian: sunt cele 2240 de zile
reale ale strategiei, cu excursia lor intraday, scalate. Scriptul isi valideaza singur
rezultatul printr-un Monte Carlo inainte sub exact aceeasi politica.

Ce face modulul asta
--------------------
NIMIC inteligent. Citeste tabelul si cauta in el. Toata gandirea s-a facut offline;
aici e doar o cautare pe grila, scrisa ca sa fie imposibil de citit gresit.

Doua decizii explicite, pentru ca ele conteaza
----------------------------------------------
1. Intre doua puncte de pe grila de echitate luam MINIMUL celor doua tinte, nu cea mai
   apropiata. Politica are praguri abrupte langa bariere (la -8% cu 20 de zile ramase
   cade de la 2.50 la 0.30). Rotunjirea in sus langa o bariera inseamna marime mare
   exact unde nu ai loc. Asimetria riscului cere rotunjire in jos.
2. Zilele ramase se numara in zile lucratoare, fara calendarul de sarbatori. Deci
   supraestimam usor timpul ramas, ceea ce face politica putin mai rabdatoare. E
   directia sigura: rabdarea nu te scoate din cont.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("live.sizing")

COLS = ("faza", "zile_ramase", "echitate", "vol_target_pct")


@dataclass
class SizingPolicy:
    """Tabelul (faza x zile_ramase x echitate) -> tinta de volatilitate zilnica, in %."""

    grid_eq: np.ndarray                 # punctele de echitate, crescator
    table: dict[int, np.ndarray]        # faza -> matrice [zile_ramase-1, echitate]
    horizon: int
    source: str

    # ---------- constructie ----------
    @classmethod
    def from_csv(cls, path: str | Path) -> "SizingPolicy":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tabelul de marime lipseste: {p}")
        d = pd.read_csv(p)
        missing = [c for c in COLS if c not in d.columns]
        if missing:
            raise ValueError(f"{p}: coloane lipsa {missing}; astept {list(COLS)}")

        grid_eq = np.sort(d["echitate"].unique())
        horizon = int(d["zile_ramase"].max())
        table: dict[int, np.ndarray] = {}
        for ph, g in d.groupby("faza"):
            m = (g.pivot_table(index="zile_ramase", columns="echitate", values="vol_target_pct")
                   .reindex(index=range(1, horizon + 1), columns=grid_eq))
            if m.isna().any().any():
                raise ValueError(f"{p}: faza {ph} are gauri in grila "
                                 f"({int(m.isna().sum().sum())} celule)")
            table[int(ph)] = m.to_numpy(dtype=float)

        if not table:
            raise ValueError(f"{p}: niciun rand")
        log.info("politica de marime: %s (faze %s, orizont %d zile, %d puncte de echitate "
                 "intre %+.1f%% si %+.1f%%)", p.name, sorted(table), horizon, len(grid_eq),
                 grid_eq[0] * 100, grid_eq[-1] * 100)
        return cls(grid_eq=grid_eq, table=table, horizon=horizon, source=str(p))

    # ---------- cautare ----------
    def vol_target_pct(self, phase: int, days_left: int, equity_rel: float) -> float:
        """
        Tinta de volatilitate zilnica, in procente (2.5 = 2.5%/zi).

        `equity_rel` e relativ la balanta de la inceputul fazei: +0.03 = esti pe plus cu 3%.
        Starile din afara grilei se taie la margine - sub -10% sau peste +10% esti oricum
        intr-o lume in care tabelul nu mai are ce sa spuna.
        """
        if phase not in self.table:
            raise KeyError(f"faza {phase} nu exista in {self.source} (am {sorted(self.table)})")
        row = int(np.clip(days_left, 1, self.horizon)) - 1
        e = float(np.clip(equity_rel, self.grid_eq[0], self.grid_eq[-1]))

        j = int(np.searchsorted(self.grid_eq, e, side="left"))
        n = len(self.grid_eq)
        lo = max(0, min(j - 1, n - 1))
        hi = max(0, min(j, n - 1))
        # Praful de virgula mobila nu are voie sa conteze: echitate/baza - 1 pentru un plus
        # de exact 5% da 0.050000000000000044, care cade la DREAPTA lui 0.05 pe grila. Fara
        # verificarea asta pe ambii vecini, un punct exact de pe grila ar fi tratat ca fiind
        # intre doua si ar lua valoarea cosului de deasupra.
        if abs(self.grid_eq[hi] - e) <= 1e-9:
            lo = hi
        elif abs(self.grid_eq[lo] - e) <= 1e-9:
            hi = lo
        m = self.table[phase]
        return float(min(m[row, lo], m[row, hi]))      # intre doua puncte, rotunjim in jos


def trading_days_left(start: str, today: str, horizon: int) -> int:
    """Zile lucratoare ramase din program. Fara calendar de sarbatori - vezi nota de sus."""
    used = int(np.busday_count(np.datetime64(str(start), "D"), np.datetime64(str(today), "D")))
    return int(np.clip(horizon - used, 1, horizon))
