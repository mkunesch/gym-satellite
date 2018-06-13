"""Satellite environments for OpenAI gym."""

from .satellite_basic import SatelliteBasic


class SatellitePerturbation(SatelliteBasic):
    """Satellite environment with initial perturbation.

    The satellite is put into the correct orbit but its radius and angular
    velocity are perturbed by 10%. The agent has to return it to the correct
    orbit.
    """
    initial_perturbation = 0.15
    start_from_rest = False
    drag_coeff = 0


class SatelliteDrag(SatelliteBasic):
    """Satellite environment with atmospheric drag.

    The satellite is put into the correct orbit but (grossly exaggerated)
    atmospheric drag leads to orbital decay. The agent has to fire the engines
    to ensure it stays in orbit.
    """
    initial_perturbation = 0
    start_from_rest = False
    drag_coeff = 0.05


class SatelliteRest(SatelliteBasic):
    """Satellite environment starting from rest.

    The satellite is put at the correct distance to Earth but almost started
    from rest (just a small angular velocity to break the symmetry). The agent
    has to fire the engines to put it into orbit.
    """
    initial_perturbation = 0
    start_from_rest = True
    drag_coeff = 0
