"""Printer and filament: can this printer print this filament? (CONTRACTS.md section 18.8).

`check(printer, filament)` is a pure function of two profiles (`topopt.printers.PrinterProfile`, `topopt.materials_catalog.FilamentProfile`).
It reads nothing else: no file, no catalog, no clock. Its answer is a fact for a person, never a stop.

Each check compares two VALUES THAT A SOURCE STATES (a printer limit and a filament need); the reason names both values and where each
comes from. There is no threshold of our own: a margin, a factor or a "close enough" would be a new number. A value that a catalog does not
state gives `unknown`, never a guess.

States of one check and of the whole answer:
- `ok`: the stated values fit.
- `care`: it can work, but the person has to do or know something (fit a hardened nozzle, fit a chamber heater, take the lower
  end of the temperature range, know the wet values of a polyamide, know that a linear model does not fit a soft filament).
  The drying setting of the maker is a text of the answer (`drying`): information, never a state.
- `no`: a stated need is above a stated limit and the printer cannot do it as it is.
- `unknown`: a value is missing on one side, so the check cannot say.
The state of the answer is the worst state of its checks: `no`, then `unknown`, then `care`, then `ok`.
"""
from __future__ import annotations

from dataclasses import dataclass

STATES = ("ok", "care", "no", "unknown")
_WORST = {"ok": 0, "care": 1, "unknown": 2, "no": 3}      # the order of the answer: no > unknown > care > ok
HARD_NOZZLES = ("hardened_steel", "tungsten_carbide")      # the stock nozzle materials of printers.NOZZLE_MATERIALS that resist fibre
CHECKS = ("printer_type", "nozzle_temperature", "bed_temperature", "enclosure", "chamber", "nozzle_wear", "extruder", "material_model",
          "moisture")


@dataclass(frozen=True)
class Reason:
    """One check: `check` is its name (`CHECKS`), `state` its state, `text` one sentence with both values and their sources."""
    check: str
    state: str
    text: str

    def to_dict(self) -> dict:
        return {"check": self.check, "state": self.state, "text": self.text}


@dataclass(frozen=True)
class Compat:
    """The answer of `check`: the state of the whole, and one reason for each check that applies."""
    state: str
    printer: str                       # the name of the printer, with its id
    filament: str                      # the name of the filament, with its id
    reasons: tuple[Reason, ...]
    drying: str | None = None          # the maker's drying setting, as text (information; it never changes the state)

    def to_dict(self) -> dict:
        return {"state": self.state, "printer": self.printer, "filament": self.filament,
                "reasons": [r.to_dict() for r in self.reasons], "drying": self.drying}

    def line(self) -> str:
        """One line for a person: the state, both names, and the checks that are not `ok`."""
        open_ = [r for r in self.reasons if r.state != "ok"]
        tail = "; ".join(r.text for r in open_) if open_ else "the stated values fit"
        return f"{self.state}: {self.filament} on {self.printer}: {tail}"


def worst(states) -> str:
    """The worst of some states (`no`, then `unknown`, then `care`, then `ok`); `unknown` for none."""
    states = list(states)
    return max(states, key=_WORST.__getitem__) if states else "unknown"


def _n(x) -> str:
    return f"{x:g}"


def _span(pair) -> str:
    lo, hi = pair
    return f"{_n(lo)} C" if lo == hi else f"{_n(lo)} to {_n(hi)} C"


def _printer_ref(p) -> str:
    return f"{p.name} ({p.origin_text})"


def _filament_ref(f) -> str:
    sheet = f", sheet {f.sheet_version}" if f.sheet_version and f.sheet_version != "none stated" else ""
    return f"{f.name} ({f.origin_text}{sheet})"


def refs(printer, filament) -> tuple[str, str]:
    """The two references that every reason writes: the printer with its source, the filament with its source and sheet. A long list
    (`materials_catalog.printer_fit`) replaces them with the names and gives the source once for each filament."""
    return _printer_ref(printer), _filament_ref(filament)


def _limit_check(name: str, label: str, p, f, limit, need) -> Reason:
    """The printer limit `limit` (one number or None) against the filament range `need` ([min, max] or None)."""
    if need is None and limit is None:
        return Reason(name, "unknown", f"{label}: neither {_filament_ref(f)} nor {_printer_ref(p)} states it")
    if need is None:
        return Reason(name, "unknown", f"{label}: {_filament_ref(f)} states no range; {_printer_ref(p)} reaches {_n(limit)} C")
    if limit is None:
        return Reason(name, "unknown", f"{label}: {_filament_ref(f)} needs {_span(need)}; {_printer_ref(p)} states no maximum")
    lo, hi = need
    both = f"{_filament_ref(f)} needs {_span(need)}; {_printer_ref(p)} reaches {_n(limit)} C"
    if limit < lo:
        return Reason(name, "no", f"{label}: {both}, which is below the lowest value")
    if limit < hi:
        return Reason(name, "care", f"{label}: {both}, so only the lower part of the range")
    return Reason(name, "ok", f"{label}: {both}")


def _enclosure(p, f) -> Reason | None:
    need, have = f.enclosure, p.enclosure
    if f.enclosed_note and have in ("enclosed", "heated"):
        # The sheet has a text about printing in an enclosed printer: the person has to know it, whatever the enclosure need is.
        return Reason("enclosure", "care", f"enclosure: {_filament_ref(f)} says: \"{f.enclosed_note}\"; {_printer_ref(p)} is {have}")
    where = f"{_filament_ref(f)} {'needs an enclosed printer' if need == 'required' else 'recommends an enclosed printer' if need == 'recommended' else 'needs no enclosure'}"
    if need is None:
        # An enclosed or heated printer meets any enclosure need, so a sheet that states none is no gap there. An open printer, or one that
        # does not say, leaves the question open.
        state = "ok" if have in ("enclosed", "heated") else "unknown"
        return Reason("enclosure", state, f"enclosure: {_filament_ref(f)} states nothing about an enclosure; {_printer_ref(p)} is {have or 'not stated'}")
    if need == "no":
        return Reason("enclosure", "ok", f"enclosure: {where}; {_printer_ref(p)} is {have or 'not stated'}")
    if have is None:
        return Reason("enclosure", "unknown", f"enclosure: {where}; {_printer_ref(p)} does not state if it is enclosed")
    if have == "open":
        return Reason("enclosure", "no" if need == "required" else "care", f"enclosure: {where}; {_printer_ref(p)} is open")
    return Reason("enclosure", "ok", f"enclosure: {where}; {_printer_ref(p)} is {have}")


def _chamber(p, f) -> Reason | None:
    """The chamber temperature that the sheet names against the chamber heater of the printer. No line if the sheet names none."""
    want = f.chamber_c_min
    if want is None:
        return None
    label = f"chamber: {_filament_ref(f)} names a chamber of {_n(want)} C or more"
    if p.enclosure == "heated" and p.chamber_max_c is not None:
        have = f"{_printer_ref(p)} heats its chamber up to {_n(p.chamber_max_c)} C"
        if p.chamber_max_c >= want:
            return Reason("chamber", "ok", f"{label}; {have}")
        return Reason("chamber", "no", f"{label}; {have}, which is below it")
    if p.enclosure == "heated":
        return Reason("chamber", "unknown", f"{label}; {_printer_ref(p)} has a chamber heater and states no maximum")
    if p.enclosure is None:
        return Reason("chamber", "unknown", f"{label}; {_printer_ref(p)} does not state if it is enclosed")
    return Reason("chamber", "care", f"{label}; {_printer_ref(p)} is {p.enclosure} and states no chamber heater")


def _wear(p, f) -> Reason | None:
    """A filament with fibre or a sheet that asks for a hardened nozzle, against the stock nozzle of the printer."""
    asks = f.hardened_nozzle is True or (f.hardened_nozzle is None and f.abrasive is True)
    if not asks:
        if f.hardened_nozzle is None and f.abrasive is None:
            return Reason("nozzle_wear", "unknown", f"nozzle: {_filament_ref(f)} does not say if it needs a hardened nozzle")
        return None
    why = "asks for a hardened nozzle" if f.hardened_nozzle is True else "holds fibre and the sheet does not say about the nozzle"
    label = f"nozzle: {_filament_ref(f)} {why}"
    stock = p.stock_nozzle_material
    if stock is None:
        return Reason("nozzle_wear", "unknown", f"{label}; {_printer_ref(p)} states no stock nozzle material")
    if stock in HARD_NOZZLES:
        return Reason("nozzle_wear", "ok", f"{label}; {_printer_ref(p)} ships with a {stock.replace('_', ' ')} nozzle")
    return Reason("nozzle_wear", "care", f"{label}; {_printer_ref(p)} ships with a {stock} nozzle, so a hardened one has to be fitted")


def _extruder(p, f) -> Reason | None:
    """A soft filament, or a sheet that recommends a direct drive extruder, against the extruder of the printer. A bowden extruder is `care`:
    the sheet recommends, and a soft filament can be printed with care."""
    direct = f.direct_drive is True
    if not direct and f.soft is not True:
        return None
    label = f"extruder: {_filament_ref(f)} " + ("recommends a direct drive extruder" if direct else "is a soft filament")
    have = p.extruder
    if have is None:
        return Reason("extruder", "unknown", f"{label}; {_printer_ref(p)} states no extruder type")
    if have == "direct":
        return Reason("extruder", "ok", f"{label}; {_printer_ref(p)} has a direct drive extruder")
    return Reason("extruder", "care", f"{label}; {_printer_ref(p)} has a bowden extruder")


def _material_model(f) -> Reason | None:
    """A soft (rubber-like) filament: the engine solves a linear elastic model, which does not fit this material. The fact is about the
    filament and does not depend on the printer."""
    if f.soft is not True:
        return None
    return Reason("material_model", "care", f"material model: {_filament_ref(f)} is a soft, rubber-like filament; a linear model does not fit "
                                            "this material")


def _moisture(f) -> Reason | None:
    """A sheet with wet values (a polyamide): the loss is a fact the person has to know. It does not depend on the printer."""
    wet = f.wet
    if wet is None:
        return None
    pairs = []
    for key, dry, unit in (("strength_xy_mpa", f.strength_xy_mpa, "strength"), ("E_xy_mpa", f.E_xy_mpa, "modulus")):
        if wet.get(key) is not None and dry is not None:
            pairs.append(f"{unit} {_n(dry)} to {_n(wet[key])} MPa")
    values = f" In the layer plane, dry to wet: {'; '.join(pairs)}." if pairs else ""
    return Reason("moisture", "care", f"moisture: {_filament_ref(f)} gives wet values. Condition: {wet.get('condition')}.{values} "
                                      "The dry values hold for a dry part only")


def check(printer, filament) -> Compat:
    """Can `printer` print `filament`? Both are profiles; the answer is a `Compat` (see the module text)."""
    reasons: list[Reason] = []
    if printer.printer_type != "fdm":
        reasons.append(Reason("printer_type", "no", f"printer type: a filament is for an fdm printer; {_printer_ref(printer)} is {printer.printer_type}"))
    reasons.append(_limit_check("nozzle_temperature", "nozzle temperature", printer, filament, printer.max_nozzle_c, filament.nozzle_c))
    reasons.append(_limit_check("bed_temperature", "bed temperature", printer, filament, printer.max_bed_c, filament.bed_c))
    for part in (_enclosure(printer, filament), _chamber(printer, filament), _wear(printer, filament), _extruder(printer, filament),
                 _material_model(filament), _moisture(filament)):
        if part is not None:
            reasons.append(part)
    return Compat(worst(r.state for r in reasons), f"{printer.name} ({printer.id})", f"{filament.name} ({filament.id})", tuple(reasons),
                  drying=filament.drying)
