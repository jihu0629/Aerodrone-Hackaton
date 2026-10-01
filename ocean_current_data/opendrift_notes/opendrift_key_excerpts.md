# OpenDrift 핵심 코드 발췌 (v1.14.11, GPL-2.0)

## 1. 해안 상호작용 → 'stranded' (opendrift/models/basemodel/__init__.py)
```python
    def interact_with_coastline(self, final=False, intermediate_simulation=False):
        """Coastline interaction according to configuration setting"""

        if self.num_elements_active() == 0:
            return
        if not hasattr(self, 'environment') or not hasattr(self.environment, 'land_binary_mask'):
            return

        i = self.get_config('general:coastline_action')
        if i == 'none':  # Do nothing
            return

        if intermediate_simulation is True:
            return  # Coastline interaction is left for next simulationstarting from saved output

        coastline_approximation_precision = self.get_config('general:coastline_approximation_precision')

        if final is True:  # Get land_binary_mask for final location
            en, en_prof, missing = \
                self.env.get_environment(['land_binary_mask'],
                                     self.time,
                                     self.elements.lon,
                                     self.elements.lat,
                                     self.elements.z,
                                     element_ID = self.elements.ID)
            self.environment.land_binary_mask = en.land_binary_mask

        if i == 'stranding':  # Deactivate elements on land, but not in air
            on_land = np.where(self.environment.land_binary_mask == 1)[0]
            if len(on_land) == 0:
                logger.debug('No elements hit coastline.')
                return

            logger.debug('%s elements hit land, moving them to the coastline.' % len(on_land))

            self.deactivate_elements(
                (self.environment.land_binary_mask == 1) & (self.elements.z <= 0),
                reason='stranded'
            )

            if not coastline_approximation_precision:
                return

            self.elements.lon[on_land], self.elements.lat[on_land] = coastline_crossing(
                self._elements_previous.lon[self.elements.ID][on_land],
                self._elements_previous.lat[self.elements.ID][on_land],
```

## 2. coastline_action 설정 정의
```python
                                        'default': '', 'level': CONFIG_LEVEL_BASIC,
                                        'description': 'Name of simulation'},
            'general:coastline_action': {
                'type': 'enum',
                'enum': ['none', 'stranding', 'previous'],
                'default': 'stranding',
                'level': CONFIG_LEVEL_BASIC,
                'description': 'None means that objects may also move over land. '
                    'stranding means that objects are deactivated if they hit land. '
                    'previous means that objects will move back to the previous location '
                    'if they hit land'
            },
            'general:coastline_approximation_precision': {
```

## 3. OceanDrift 요소 속성·필요 변수·설정 (opendrift/models/oceandrift.py)
```python
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED

# Defining the oil element properties
class Lagrangian3DArray(LagrangianArray):
    """Extending LagrangianArray for elements moving in 3 dimensions
    The Particle may be buoyant and/or subject to vertical mixing
    buoyant bahaviour is described by terminal velocity
    """

    variables = LagrangianArray.add_variables([
        ('wind_drift_factor', {'dtype': np.float32,
                               'units': '1',
            'description': 'Elements at surface are moved with this '
                'fraction of the vind vector, in addition to currents '
                'and Stokes drift',
                               'default': 0.02}),
        ('current_drift_factor', {'dtype': np.float32,
                                  'units': '1',
            'description': 'Elements are moved with this fraction of the '
                            'current vector, in addition to currents '
                            'and Stokes drift',
                               'default': 1}),
        ('terminal_velocity', {'dtype': np.float32,
                               'units': 'm/s',
            'description': 'Terminal rise/sinking velocity (buoyancy) '
                'in the ocean column',
                               'default': 0.})])


class OceanDrift(OpenDriftSimulation):
    """Open source buoyant particle trajectory model based on OpenDrift.

        Developed at MET Norway

        Generic module for particles that move in 3 dimensions
        and may be to vertical turbulent mixing
        with the possibility for positive or negative buoyancy

        Particles could be e.g. oil droplets, plankton, nutrients or sediments,
        Model may be subclassed for more specific behaviour.

    """

    ElementType = Lagrangian3DArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0,
            'store_previous_if': ['drift:vertical_advection', 'is', True]},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0,
            'skip_if': ['drift:vertical_advection', 'is', False]},
        'ocean_vertical_diffusivity': {'fallback': 0,
             'skip_if': ['drift:vertical_mixing', 'is', False],
             'profiles': True},
        'horizontal_diffusivity': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0,
            'skip_if': ['drift:stokes_drift', 'is', False]},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0,
            'skip_if': ['drift:stokes_drift', 'is', False]},
        'ocean_mixed_layer_thickness': {
            'fallback': 50, 'skip_if': ['drift:vertical_mixing', 'is', False]},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
        'land_binary_mask': {'fallback': None},
        }


    def __init__(self, *args, **kwargs):

        if 'machine_learning_dict' in kwargs:
            logger.debug('Machine learning correction supplied.')
            mld = kwargs['machine_learning_dict']
            del kwargs['machine_learning_dict']
```
```python
                'enum': ['environment', 'stepfunction', 'windspeed_Sundby1983',
                 'windspeed_Large1994', 'constant'], 'level': CONFIG_LEVEL_ADVANCED,
                 'units': 'seconds', 'description': 'Algorithm/source used for profile of vertical diffusivity. Environment means that diffusivity is aquired from readers or environment constants/fallback.'},
            'vertical_mixing:background_diffusivity': {'type': 'float', 'min': 0, 'max': 1, 'default': 1.2e-5,
                'level': CONFIG_LEVEL_ADVANCED, 'units': 'm2s-1', 'description':
                'Background diffusivity used below mixed layer for wind parameterisations.'},
            'vertical_mixing:TSprofiles': {'type': 'bool', 'default': False, 'level':
                CONFIG_LEVEL_ADVANCED,
                'description': 'Update T and S profiles within inner loop of vertical mixing. This takes more time, but may be slightly more accurate.'},
            'drift:wind_drift_depth': {'type': 'float', 'default': 0.1,
                'min': 0, 'max': 10, 'units': 'meters',
                'description': 'The direct wind drift (windage) is linearly decreasing from the surface value (wind_drift_factor) until 0 at this depth.',
                'level': CONFIG_LEVEL_ADVANCED},
            'drift:stokes_drift': {'type': 'bool', 'default': True,
                'description': 'Advection elements with Stokes drift (wave orbital motion).',
                'level': CONFIG_LEVEL_ADVANCED},
            'drift:stokes_drift_profile': {'type': 'enum', 'default': 'Phillips',
                                           'enum': ['monochromatic', 'exponential', 'Phillips', 'windsea_swell'],
                                           'description': 'Algorithm to calculate Stokes drift at depth from surface value',
                                           'level': CONFIG_LEVEL_ADVANCED},
            'drift:use_tabularised_stokes_drift': {'type': 'bool', 'default': False,
                'description': 'If True, Stokes drift is estimated from wind based on look-up-tables for given fetch (drift:tabularised_stokes_drift_fetch).',
                'level': CONFIG_LEVEL_ADVANCED},
            'drift:tabularised_stokes_drift_fetch': {'type': 'enum', 'enum': ['5000', '25000', '50000'], 'default': '25000',
                'level': CONFIG_LEVEL_ADVANCED, 'description':
                'The fetch length when using tabularised Stokes drift.'},
            'general:seafloor_action': {'type': 'enum', 'default': 'lift_to_seafloor',
                'enum': ['none', 'lift_to_seafloor', 'deactivate', 'previous'],
                'description': '"deactivate": elements are deactivated; "lift_to_seafloor": elements are lifted to seafloor level; "previous": elements are moved back to previous position; "none"; seafloor is ignored.',
                'level': CONFIG_LEVEL_ADVANCED},
            'drift:truncate_ocean_model_below_m': {'type': 'float', 'default': None,
                'min': 0, 'max': 10000, 'units': 'm',
                'description': 'Ocean model data are only read down to at most this depth, and extrapolated below. May be specified to read less data to improve performance.',
                'level': CONFIG_LEVEL_ADVANCED},
             'seed:z': {'type': 'float', 'default': 0,
                    'min': -10000, 'max': 0, 'units': 'm',
```

## 4. 수평 이류·확산 (physics_methods / oceandrift advect)
```python
```
