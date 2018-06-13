from gym.envs.registration import register

register(
    id='SatellitePerturbation-v0',
    entry_point='satellite.envs:SatellitePerturbation',
    timestep_limit=5000,
    nondeterministic = True
)

register(
    id='SatelliteDrag-v0',
    entry_point='satellite.envs:SatelliteDrag',
    timestep_limit=5000,
    nondeterministic = True
)

register(
    id='SatelliteRest-v0',
    entry_point='satellite.envs:SatelliteRest',
    timestep_limit=5000,
    nondeterministic = True
)
