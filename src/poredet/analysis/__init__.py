"""Bewertung auf Basis der Messwerte.

Die Filter sind Plugins mit Begruendungspflicht: jede verworfene Pore behaelt den Grund,
aus dem sie verworfen wurde, und wird nicht geloescht, sondern mitgefuehrt. Das ist die
Grundlage fuer die haeufigste Frage am Kontrollbild - "warum fehlt diese Pore?".

Der Import von :mod:`geometry` steht hier, weil er die geometrischen Filter in der
Registry anmeldet. Ohne ihn waeren sie zwar geschrieben, aber unbekannt.
"""

from . import geometry  # noqa: F401  - registriert die geometrischen Filter
from .filters import FilterContext, apply, available, register

__all__ = ["FilterContext", "apply", "available", "register", "geometry"]
