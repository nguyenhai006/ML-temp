"""Spatial Galerkin twin. Only public geometry, generator constants and observations.

Use bilinear functions on coarser tiles, with separate functions for each zone
and wall material/layer. Project the simulator's face resistances, storage,
slab exchange, diffuser weights and zone sensors. Zone means, return mixing,
coil output and boiler temperature are computed from the twin's own state.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from smart_buildings.smart_control.simulator import building, constants, finite_difference_solver
from smart_buildings.smart_control.utils.scenario_generator import MaterialsConfig


def hat_basis(rows, cols, labels, stride):
    """Bilinear interpolation with a partition of unity within each label."""
    nodes, weights = [], []
    fr, fc = (rows % stride) / stride, (cols % stride) / stride
    width = int(cols.max() // stride + 2)
    height = int(rows.max() // stride + 2)
    for dr, wr in [(0, 1 - fr), (1, fr)]:
        for dc, wc in [(0, 1 - fc), (1, fc)]:
            nodes.append((labels * height + rows // stride + dr) * width + cols // stride + dc)
            weights.append(wr * wc)
    weights = np.concatenate(weights)
    nz = weights > 0
    _, indices = np.unique(np.concatenate(nodes)[nz], return_inverse=True)
    return sparse.csr_matrix((weights[nz], (np.tile(np.arange(len(rows)), 4)[nz], indices)))


class Geometry:
    """Build the stock stencil once, keeping its material-dependent factors."""
    def __init__(self, meta):
        defaults = MaterialsConfig()
        def material(key):
            return building.MaterialProperties(
                *(getattr(defaults, key + "_" + prop)
                  for prop in ("conductivity", "heat_capacity", "density")))
        # Construct a nominal unscoped stock grid with all physical inputs
        # explicit. This path never parses gin, loads weather or seeds the
        # shared convection RNG. It can coexist with the participant's env.
        grid = building.FloorPlanBasedBuilding(
            cv_size_cm=20, floor_height_cm=300., initial_temp=294.,
            inside_air_properties=material("interior_cv"),
            inside_wall_properties=material("interior_wall_cv"),
            building_exterior_properties=material("exterior_cv"),
            interior_mass_properties=material("interior_mass_cv"),
            zone_map=meta["zone_map"], floor_plan=meta["floor_plan"])
        solver = finite_difference_solver.FiniteDifferenceSolver(grid, 300.)
        self.cells = solver._solved
        self.shape = solver._shape
        self.rows, self.cols = np.unravel_index(self.cells, self.shape)
        self.kind = np.zeros(len(self.cells), int)
        conductivity = solver._conductivity.ravel()[self.cells]
        self.kind[conductivity == defaults.interior_wall_cv_conductivity] = 1
        self.kind[conductivity == defaults.exterior_cv_conductivity] = 2
        self.dx, self.dy = [x.ravel()[self.cells] for x in (solver._horizontal, solver._vertical)]
        self.area = self.dx * self.dy
        self.faces = [x.ravel()[self.cells] for x in solver._ambient_face_areas]
        self.active_mass = solver._mass_coupling > 0
        self.names = sorted(meta["zone_action_indices"])
        points = [np.asarray(grid.room_dict[v.replace("VAV_", "room_")]).T for v in self.names]
        self.room_size = np.array([p.shape[1] for p in points])
        rr, cc = np.hstack(points)
        flat = np.ravel_multi_index((rr, cc), self.shape)
        solved = np.searchsorted(self.cells, flat)
        assert np.array_equal(self.cells[solved], flat)
        rows = np.repeat(np.arange(len(self.names)), self.room_size)
        shape = (len(self.names), len(self.cells))
        self.sensed = sparse.csr_matrix((np.repeat(1. / self.room_size, self.room_size),
                                         (rows, solved)), shape=shape)
        self.applied = sparse.csc_matrix((grid.diffusers[rr, cc], (solved, rows)), shape=shape[::-1])
        self.applied.eliminate_zeros()
        # Read only the stencil connectivity from the nominal object. Shared
        # face areas and both half-cell resistances are recalculated per draw.
        upper = sparse.triu(solver._conduction, k=1).tocoo()
        i, j = upper.row, upper.col
        along_col = self.rows[i] == self.rows[j]
        self.face_area = 3. * np.where(along_col, np.minimum(self.dy[i], self.dy[j]),
                                      np.minimum(self.dx[i], self.dx[j]))
        self.dist_i = np.where(along_col, self.dx[i], self.dy[i]) / 2
        self.dist_j = np.where(along_col, self.dx[j], self.dy[j]) / 2
        self.i, self.j = i, j
        # Generator boundaries are adiabatic at missing neighbours, with
        # exposed faces coupled by film, so no eliminated-neighbour term.
        assert np.max(abs(solver._dirichlet_conductance)) < 1e-9
        # Recover zone labels from the sensing sparsity, not zone numbering
        # assumptions. Enlarged wall cells can still belong to a sensed zone.
        sensor = self.sensed.tocoo()
        zone = np.zeros(len(self.cells), int)
        zone[sensor.col] = sensor.row + 1
        self.labels = (zone * 3 + self.kind) * 4 + (self.dx < .2) * 2 + (self.dy < .2)
        self.meta = meta


class Mesh:
    """Projected operators reused across parameter trials on the same mesh."""
    def __init__(self, geo, stride):
        self.geo = geo
        basis = hat_basis(geo.rows, geo.cols, geo.labels, stride)
        self.size = basis.shape[1]
        self.sensed = (geo.sensed @ basis).tocsr()
        self.diffusers = (basis.T @ geo.applied).tocsc()
        # The known room load is uniform. Only the implicit air term uses
        # diffuser shares in the shipped finite-difference implementation.
        self.uniform = self.sensed.T.tocsr()

        def diagonal(weights):
            return (basis.T @ basis.multiply(weights[:, None])).tocsc()
        self.storage = [diagonal(geo.area * 3 / 300 * (geo.kind == k)) for k in range(3)]
        self.slab = diagonal(geo.area * geo.active_mass)
        self.film = []
        for face, thickness in zip(geo.faces, (geo.dx, geo.dy)):
            for k in range(3):
                for dx in np.unique(thickness[(face > 0) & (geo.kind == k)]):
                    mask = (geo.kind == k) & (thickness == dx)
                    self.film.append((k, dx, diagonal(face * mask)))
        self.edges = []
        difference = basis[geo.i] - basis[geo.j]
        keys = np.column_stack((geo.kind[geo.i], geo.kind[geo.j], geo.dist_i, geo.dist_j))
        for (a, b, da, db) in np.unique(keys, axis=0):
            keep = (geo.kind[geo.i] == a) & (geo.kind[geo.j] == b)
            keep &= (geo.dist_i == da) & (geo.dist_j == db)
            block = difference[keep]
            self.edges.append((int(a), int(b), da, db,
                               (block.T @ block.multiply(geo.face_area[keep, None])).tocsc()))

    def solver(self, materials, h):
        k = [50., materials["interior_wall_cv_conductivity"], materials["exterior_cv_conductivity"]]
        cap = [1.2 * 1005, materials["interior_wall_cv_density"] * 900.,
               materials["exterior_cv_density"] * materials["exterior_cv_heat_capacity"]]
        storage = sum(c * block for c, block in zip(cap, self.storage))
        c_slab = 3 * materials["interior_mass_cv_density"] * 1000 / 300
        g_slab = 15 / 3
        retention = c_slab / (c_slab + g_slab)
        slab = self.slab * (g_slab * retention)
        exposure = sum(h / (1 + h * d / (2 * k[a])) * block for a, d, block in self.film)
        conduction = sum(block / (da / k[a] + db / k[b]) for a, b, da, db, block in self.edges)
        mat = (conduction + storage + slab + exposure).tocsc()
        # Unsupported linear combinations on a one-cell wall line have zero
        # mass. Tiny diagonal tie to the reference temperature selects a unique
        # solution without changing the physical transfers at sensor precision.
        mat += sparse.eye(self.size, format="csc") * 1e-9
        lu = splu(mat)
        response = np.ascontiguousarray(lu.solve(self.diffusers.toarray()))
        return Dynamics(self, storage.tocsr(), slab.tocsr(), retention,
                        np.asarray(exposure.sum(axis=1)).ravel(), lu, response)


class Dynamics:
    def __init__(self, mesh, storage, slab, retention, exposure, lu, response):
        self.storage, self.slab, self.retention = storage, slab, retention
        self.exposure, self.lu, self.response = exposure, lu, response
        self.sensed, self.uniform = mesh.sensed, mesh.uniform
        self.cross = self.sensed @ response
        self.eye = np.eye(len(self.cross))
        self.reset()

    def reset(self):
        self.air = np.zeros_like(self.exposure)  # Kelvin relative to 294
        self.mass = np.zeros_like(self.air)
        self.zones = self.sensed @ self.air + 294.
        self.tank = 340.

    def step(self, oat, gains, air_w_per_k, supply, coil_w):
        rhs = (self.storage @ self.air + self.slab @ self.mass +
               self.exposure * (oat - 294.) + self.uniform @ (gains + coil_w))
        passive = self.lu.solve(rhs)
        power = np.linalg.solve(self.eye + air_w_per_k[:, None] * self.cross,
                                air_w_per_k * (supply - 294. - self.sensed @ passive))
        self.air = passive + self.response @ power
        self.mass = self.retention * self.mass + (1 - self.retention) * self.air
        self.zones = self.sensed @ self.air + 294.


class Inputs:
    """Vectorize public action normalization. Exogenous inputs are measured at t."""
    def __init__(self, geo, continuous, discrete, oat, comfort, occupancy):
        m = geo.meta
        norm = m["action_normalization_metadata"]
        def native(ix, spec):
            s = norm[spec]
            x = continuous[:, ix].astype(float)
            f = (x - s["min_normalized_value"]) / (s["max_normalized_value"] - s["min_normalized_value"])
            return np.clip(f, 0, 1) * (s["max_native_value"] - s["min_native_value"]) + s["min_native_value"]
        ids = [f"ahu_{i}" for i in range(1, len(m["ahu_zone_assignments"]) + 1)]
        self.ahu = np.array([next(i for i, a in enumerate(ids)
                             if v.replace("VAV_", "room_") in m["ahu_zone_assignments"][a])
                             for v in geo.names])
        groups = [(np.flatnonzero(self.ahu == i)) for i in range(len(ids))]
        damp, valve = "supply_air_damper_percentage_command", "reheat_valve_percentage_command"
        damper = np.maximum(.1, native([m["zone_action_indices"][v][damp] for v in geo.names], damp))
        valve = native([m["zone_action_indices"][v][valve] for v in geo.names], valve)
        self.run = discrete[:, :-1]
        self.recirc = np.minimum(.7, native(m["recirculation_indices"], ids[0] + "_recirculation_percentage_command"))
        self.sat = native(m["temp_indices"], ids[0] + "_supply_air_temperature_setpoint")
        pressure = native(m["pressure_indices"][:len(ids)], ids[0] + "_supply_air_static_pressure_setpoint")
        pressure = np.maximum(200, pressure) * self.run
        self.flow = np.maximum(1e-5, 2 * damper * np.sqrt(pressure[:, self.ahu] / 2000))
        self.weights = self.flow.copy()
        for idx in groups:
            total = np.sum(self.flow[:, idx], axis=1, keepdims=True)
            self.weights[:, idx] /= total
            self.flow[:, idx] *= np.minimum(1., 20 / total)
        self.air_capacity = constants.AIR_DENSITY * constants.AIR_HEAT_CAPACITY * self.flow
        water_pressure = native(m["pressure_indices"][-1], "differential_pressure_setpoint") * discrete[:, -1]
        self.water_capacity = 4180e3 * .108 / 3600 * valve * np.sqrt(water_pressure[:, None])
        self.tank_setpoint = native(m["water_indices"][0], "supply_water_temperature_setpoint")
        self.gains = ((.2 + .8 * comfort[:, None]) * 10 * .04 * geo.room_size +
                      70 * occupancy[:, None] / len(geo.names))
        self.oat = oat
        self.tank_ua = 4 * np.pi / (np.log(.56 / .5) / .067 + 1 / (5.6 * .56))

    def run_steps(self, solver, count, start=0, reset=True):
        if reset:
            solver.reset()
        predicted = np.empty((count, len(solver.zones)))
        for j, i in enumerate(range(start, start + count)):
            predicted[j] = solver.zones  # post-action VAV sensor samples pre-solve field
            ret = np.bincount(self.ahu, self.weights[i] * solver.zones)
            mixed = self.recirc[i] * ret + (1 - self.recirc[i]) * self.oat[i]
            supply = np.where(self.run[i], np.minimum(self.sat[i], mixed), mixed)[self.ahu]
            g = self.air_capacity[i]
            coil = np.minimum(g, self.water_capacity[i]) * (solver.tank - supply)
            coil = np.minimum(coil, g * np.maximum(0., 308.15 - supply))
            solver.step(self.oat[i], self.gains[i], g, supply, coil)
            loss = self.tank_ua * max(0., solver.tank - self.oat[i])
            draw = coil.sum()
            capacity = 1.5 * 4180e3
            need = capacity * (self.tank_setpoint[i] - solver.tank) / 300 + draw + loss
            burner = np.clip(need, 0., max(draw, 0.) + loss + capacity * .5 / 60)
            solver.tank += (burner - draw - loss) * 300 / capacity
        return predicted
