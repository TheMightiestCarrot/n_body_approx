import os
import numpy as np
import matplotlib.pyplot as plt
import time


class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0., dim=3, delta_t=0.001):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = delta_t
        self._max_F = 0.1 / self._delta_T
        self.dim = dim

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[2, :],
                                       loc_next[2, :]).reshape(1, n, n)
                 ))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                # loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n, n)
                     ))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5, interaction_strength=1., noise_var=0., dim=3,
                 delta_t=0.001):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.loc_std = loc_std * (float(n_balls) / 5.) ** (1 / 3)
        print(self.loc_std)
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = delta_t
        self._max_F = 0.1 / self._delta_T
        self.dim = dim

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[2, :],
                                       loc_next[2, :]).reshape(1, n, n)))).sum(axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                # loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n, n)
                     ))).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges, charges


class GravitySim(object):
    def __init__(self, n_balls=100, loc_std=1, vel_norm=0.5, interaction_strength=1, noise_var=0, dt=0.001,
                 softening=0.1, dim=3):
        self.n_balls = n_balls
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.dt = dt
        self.softening = softening

        self.dim = dim

    @staticmethod
    def compute_acceleration(pos, mass, G, softening):
        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r^3 for all particle pairwise particle separations
        inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

        ax = G * (dx * inv_r3) @ mass
        ay = G * (dy * inv_r3) @ mass
        az = G * (dz * inv_r3) @ mass

        # pack together the acceleration components
        a = np.hstack((ax, ay, az))
        return a

    def simulate_step(self, pos, vel, acc, mass):
        # (1/2) kick
        vel += acc * self.dt / 2.0

        # drift
        pos += vel * self.dt

        # update accelerations
        acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

        # (1/2) kick
        vel += acc * self.dt / 2.0

        return pos, vel, acc

    def sample_trajectory(self, T=10000, sample_freq=10, og_pos_save=None, og_vel_save=None, og_force_save=None):
        assert (T % sample_freq == 0)

        T_save = int(T / sample_freq)

        N = self.n_balls

        pos_save = np.zeros((T_save, N, self.dim))
        vel_save = np.zeros((T_save, N, self.dim))
        force_save = np.zeros((T_save, N, self.dim))

        mass = np.ones((N, 1))
        if og_pos_save is None:
            # Specific sim parameters
            pos = np.random.randn(N, self.dim)  # randomly selected positions and velocities
            vel = np.random.randn(N, self.dim)

            # Convert to Center-of-Mass frame
            vel -= np.mean(mass * vel, 0) / np.mean(mass)

        else:
            pos = np.copy(og_pos_save[-1])
            vel = np.copy(og_vel_save[-1])

        # calculate initial gravitational accelerations
        acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

        if og_pos_save is not None:
            pos, vel, acc = self.simulate_step(pos, vel, acc, mass)

        counter = 0

        for i in range(T):
            if i % sample_freq == 0:
                pos_save[counter] = pos
                vel_save[counter] = vel
                force_save[counter] = acc * mass
                counter += 1

            pos, vel, acc = self.simulate_step(pos, vel, acc, mass)

        # Add noise to observations
        pos_save += np.random.randn(T_save, N, self.dim) * self.noise_var
        vel_save += np.random.randn(T_save, N, self.dim) * self.noise_var
        force_save += np.random.randn(T_save, N, self.dim) * self.noise_var

        if og_pos_save is not None:
            pos_save = np.concatenate((og_pos_save, pos_save), axis=0)
            vel_save = np.concatenate((og_vel_save, vel_save), axis=0)
            force_save = np.concatenate((og_force_save, force_save), axis=0)

        return pos_save, vel_save, force_save, mass

    @staticmethod
    def compute_force(pos, mass, G, softening, batch_size=None):
        if batch_size is None:
            return GravitySim.compute_acceleration(pos, mass, G, softening) * mass

        rows = pos.shape[0]

        # Ensure batch_size is a divisor of num_particles, else process as a single batch
        if rows % batch_size != 0:
            raise ValueError(f"batch_size {batch_size} is not a divisor of the number of particles {rows}.")

        # Calculate the number of bodies per batch
        num_bodies = rows // batch_size

        # Reshape pos and mass to separate the batches
        pos = pos.reshape(batch_size, num_bodies, -1)
        mass = mass.reshape(batch_size, num_bodies, -1)

        # Prepare to store acceleration for all batches
        acceleration = np.zeros_like(pos)

        for b in range(batch_size):
            batch_acceleration = GravitySim.compute_acceleration(pos[b], mass[b], G, softening)
            acceleration[b] = batch_acceleration

        acceleration = acceleration * mass.reshape(batch_size, num_bodies, 1)
        return acceleration.reshape(-1, 3)

    def _energy(self, pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

        # Potential Energy:

        # positions r = [x,y,z] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        z = pos[:, 2:3]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # matrix that stores 1/r for all particle pairwise particle separations
        inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + self.softening ** 2)
        inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

        return KE, PE, KE + PE

    def plot_energies(self, loc, vel, mass):
        energies = [self._energy(loc[i, :, :], vel[i, :, :], mass, self.interaction_strength) for i in
                    range(loc.shape[0])]

        energies_array = np.array(energies)

        times = np.arange(energies_array.shape[0])

        plt.figure(figsize=(10, 6))
        plt.plot(times, energies_array[:, 0], label='Kinetic Energy', color='red')
        plt.plot(times, energies_array[:, 1], label='Potential Energy', color='blue')
        plt.plot(times, energies_array[:, 2], label='Total Energy', color='black')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy vs Time (Filtered)')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_trajectory_static(loc):
        num_dims = loc.shape[2]
        n_balls = loc.shape[1]
        if num_dims == 2:
            plt.figure(figsize=(10, 8))
            for n in range(n_balls):
                plt.plot(loc[:, n, 0], loc[:, n, 1], label=f'Particle {n + 1}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
        elif num_dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for n in range(n_balls):
                ax.plot(loc[:, n, 0], loc[:, n, 1], loc[:, n, 2], label=f'Particle {n + 1}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
        else:
            raise ValueError("Dimensions not supported for plotting")

        plt.title('Particle Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_histograms(loc, vel):
        num_dims = loc.shape[2]
        dim_labels = ['x', 'y', 'z'][:num_dims]  # Labels for dimensions
        colors = ['red', 'green', 'blue'][:num_dims]  # Color for each dimension

        plt.figure(figsize=(10, 5))

        # Positions
        plt.subplot(1, 2, 1)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            plt.hist(loc[:, :, i].flatten(), bins=20, alpha=0.5, color=color, label=f'{label}')
        plt.title('Positions')
        plt.legend()

        # Velocities
        plt.subplot(1, 2, 2)
        for i, (color, label) in enumerate(zip(colors, dim_labels)):
            plt.hist(vel[:, :, i].flatten(), bins=20, alpha=0.5, color=color, label=f'{label}')
        plt.title('Velocities')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_energy_distribution(self, loc, vel, mass, bins=50):
        energies = [self._energy(loc[i, :, :], vel[i, :, :], mass, self.interaction_strength) for i in
                    range(loc.shape[0])]
        energies_array = np.array(energies)

        plt.figure(figsize=(15, 5))

        energy_types = ['Kinetic Energy', 'Potential Energy', 'Total Energy']
        colors = ['red', 'blue', 'black']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.hist(energies_array[:, i], bins=bins, color=colors[i], alpha=0.7)
            plt.xlabel('Energy')
            plt.ylabel('Frequency')
            plt.title(f'{energy_types[i]} Histogram')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def interactive_trajectory_plot_all_particles_3d(actual_pos=None, predicted_pos=None, particle_index=None,
                                                     boxSize=1,
                                                     dims=3,
                                                     offline_plot=False, loggers=[],
                                                     video_tag="trajectories all particles 3D", trace_length=10,
                                                     alpha=0.2):
        import matplotlib
        from matplotlib.widgets import Slider
        from matplotlib.lines import Line2D
        import traceback
        og_backend = matplotlib.get_backend()
        # if we're running in headless mode, offline_plot should be True
        if (os.environ.get('DISPLAY', '') == '') and (os.name != 'nt'):
            print('running in headless mode, setting offline_plot to True')
            offline_plot = True

        try:
            if offline_plot:
                matplotlib.use('Agg')
            else:
                try:
                    matplotlib.use('Qt5Agg')
                except (NameError, KeyError):
                    matplotlib.use('TkAgg')

            if actual_pos is None and predicted_pos is None:
                raise ValueError("At least one of actual_pos or predicted_pos must be provided.")

            actual_lines, predicted_lines = [], []

            skip_steps = 2
            plt.ion()
            fig = plt.figure(figsize=(12, 8))

            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-boxSize, boxSize)
            ax.set_ylim(-boxSize, boxSize)
            ax.set_zlim(-boxSize, boxSize)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_title('3D Trajectories of all particles ' + (
                '' if particle_index is None else f'with predicted particle {str(particle_index)}'))

            plt.subplots_adjust(bottom=0.25)
            if actual_pos is not None:
                actual_lines = [ax.plot([], [], [], 'b-', label='Actual', alpha=alpha)[0] for _ in
                                range(actual_pos.shape[1])]

            if predicted_pos is not None:
                if particle_index is None:
                    num_predicted_particles = predicted_pos.shape[1]
                    predicted_lines = [ax.plot([0], [0], [0], 'r-', alpha=alpha)[0] for _ in
                                       range(num_predicted_particles)]
                else:
                    predicted_line, = ax.plot([], [], [], 'r-', label='Predicted', alpha=alpha)

            from matplotlib.lines import Line2D
            legend_lines = [Line2D([0], [0], color='blue', lw=2, label='Actual')]
            if predicted_pos is not None:
                legend_lines.append(Line2D([0], [0], color='red', lw=2, label='Predicted'))

            ax.legend(handles=legend_lines)

            max_actual_index = 0
            if actual_pos is not None:
                max_actual_index = actual_pos.shape[0] - 1
            if predicted_pos is not None:
                max_predicted_steps = predicted_pos.shape[0] - 1
            else:
                max_predicted_steps = 0

            # markers for first position
            current_actual_markers = []
            if actual_pos is not None:
                current_actual_markers = [ax.scatter([], [], [], color='blue', marker='o', s=3, depthshade=False) for _
                                          in
                                          range(actual_pos.shape[1])]

            if predicted_pos is not None:
                current_predicted_markers = [ax.scatter([], [], [], color='red', marker='o', s=3, depthshade=False) for
                                             _ in range(predicted_pos.shape[1])]
            else:
                current_predicted_markers = []

            slider_max = max(max_actual_index, max_predicted_steps)

            if not offline_plot:
                ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
                slider = Slider(ax_slider, 'Time Step', 0, slider_max, valinit=0, valfmt='%0.0f')

            def on_key(event):
                if event.key == 'left':
                    new_val = max(0, slider.val - 1)
                    slider.set_val(new_val)
                elif event.key == 'right':
                    new_val = min(slider_max, slider.val + 1)
                    slider.set_val(new_val)

            def update(val):
                if offline_plot:
                    time_step = int(val) * skip_steps
                else:
                    time_step = int(val)

                start_step = max(0, time_step - trace_length)
                last_actual_step = min(time_step, max_actual_index)
                last_predicted_step = min(time_step, max_predicted_steps)

                if actual_pos is not None:
                    for pi in range(actual_pos.shape[1]):
                        actual_lines[pi].set_data(actual_pos[start_step:last_actual_step + 1, pi, 0],
                                                  actual_pos[start_step:last_actual_step + 1, pi, 1])
                        actual_lines[pi].set_3d_properties(actual_pos[start_step:last_actual_step + 1, pi, 2])

                        # marker
                        current_actual_markers[pi]._offsets3d = (
                            actual_pos[last_actual_step, pi, 0:1], actual_pos[last_actual_step, pi, 1:2],
                            actual_pos[last_actual_step, pi, 2:3])

                # Update predicted particle(s)
                if predicted_pos is not None:
                    if particle_index is not None:
                        # Adjust predicted_line to plot up to the last predicted step
                        predicted_line.set_data(predicted_pos[start_step:last_predicted_step + 1, particle_index, 0],
                                                predicted_pos[start_step:last_predicted_step + 1, particle_index, 1])
                        predicted_line.set_3d_properties(
                            predicted_pos[start_step:last_predicted_step + 1, particle_index, 2])

                    else:
                        # Adjust each predicted particle to plot up to the last predicted step
                        for pi in range(predicted_pos.shape[1]):
                            predicted_lines[pi].set_data(predicted_pos[start_step:last_predicted_step + 1, pi, 0],
                                                         predicted_pos[start_step:last_predicted_step + 1, pi, 1])
                            predicted_lines[pi].set_3d_properties(
                                predicted_pos[start_step:last_predicted_step + 1, pi, 2])
                            current_predicted_markers[pi]._offsets3d = (
                                predicted_pos[last_predicted_step, pi, 0:1],
                                predicted_pos[last_predicted_step, pi, 1:2],
                                predicted_pos[last_predicted_step, pi, 2:3])

                if offline_plot:
                    return actual_lines + (
                        [predicted_line] if predicted_pos is not None and particle_index is not None else [])
                else:
                    fig.canvas.draw_idle()

            if not offline_plot:
                fig.canvas.mpl_connect('key_press_event', on_key)

                update(0)
                slider.on_changed(update)
                print('Showing plot, you might need to bring the plot window in focus')
                plt.show(block=True)

            if offline_plot:
                import tempfile
                from matplotlib.animation import FuncAnimation
                fps = 20
                frames = actual_pos.shape[0] // skip_steps
                anim = FuncAnimation(fig, update, frames=frames, blit=False)
                
                if loggers:
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp:
                        filename = temp.name
                        anim.save(filename, fps=fps,
                                  extra_args=['-vcodec', 'libx264', '-preset', 'fast', '-crf', '22'])

                        for logger in loggers:
                            logger.log_video(f"{video_tag}", filename, fps=fps)
                else:
                    filename = f"{video_tag}.mp4"
                    anim.save(filename, fps=fps,
                              extra_args=['-vcodec', 'libx264', '-preset', 'fast', '-crf', '22'])

        except KeyboardInterrupt:
            pass

        matplotlib.use(og_backend)

    @staticmethod
    def interactive_plotly_offline_plot(actual_pos=None, predicted_pos=None, output_file='3D_offline_plot.html', duration=8):
        import plotly.graph_objects as go
        import numpy as np

        actual = actual_pos
        predicted = predicted_pos

        particles = actual.shape[1]
        steps = actual.shape[0]

        # Initialize the figure
        # Creating initial plot data with both particles' initial positions
        fig = go.Figure(
            layout=go.Layout(
                template='plotly_white',
                updatemenus=[dict(type='buttons', showactive=False,
                                  y=0,
                                  x=1.05,
                                  xanchor='right',
                                  yanchor='top',
                                  pad=dict(t=0, r=10),
                                  buttons=[dict(label='Play',
                                                method='animate',
                                                args=[None, {'frame': {'duration': duration, 'redraw': True},
                                                             'fromcurrent': True,
                                                             'mode': 'immediate',
                                                             'transition': {'duration': duration}}]),
                                           dict(label='Stop',
                                                method='animate',
                                                args=[[None], {'frame': {'duration': 0, 'redraw': True},
                                                               'mode': 'immediate',
                                                               'transition': {'duration': 0}}])])],
                sliders=[dict(steps=[dict(method='animate',
                                          args=[[f'frame{k}'],
                                                {'mode': 'immediate', 'frame': {'duration': duration, 'redraw': True},
                                                 'fromcurrent': True}],
                                          label=f'{k}') for k in range(steps)],
                              transition={'duration': duration},
                              x=0,
                              y=0,
                              currentvalue={'font': {'size': 12}, 'prefix': 'Point: ', 'visible': True},
                              len=1.0)]
            )
        )
        # fig.update_layout(transition = {'duration': 50})
        for i in range(particles):
            # Initial lines for trajectory visualization
            fig.add_trace(go.Scatter3d(x=[actual[0, i, 0]], y=[actual[0, i, 1]], z=[actual[0, i, 2]],
                                       mode='markers+lines', marker=dict(size=5, color='blue')))
            fig.add_trace(go.Scatter3d(x=[predicted[0, i, 0]], y=[predicted[0, i, 1]], z=[predicted[0, i, 2]],
                                       mode='markers+lines', marker=dict(size=5, color='red')))
            # Placeholder markers for the current step
            fig.add_trace(go.Scatter3d(x=[actual[0, i, 0]], y=[actual[0, i, 1]], z=[actual[0, i, 2]],
                                       mode='markers', marker=dict(size=2, color='darkblue')))
            fig.add_trace(go.Scatter3d(x=[predicted[0, i, 0]], y=[predicted[0, i, 1]], z=[predicted[0, i, 2]],
                                       mode='markers', marker=dict(size=2, color='darkred')))

        # Create frames for each step
        frames = []
        for k in range(1, steps):
            start_idx = max(0, k - 15)
            frame_data = []
            for i in range(particles):
                # Actual trajectory segment
                frame_data.append(
                    go.Scatter3d(x=actual[start_idx:k, i, 0], y=actual[start_idx:k, i, 1], z=actual[start_idx:k, i, 2],
                                 mode='lines', line=dict(width=5, color='blue'), opacity=0.3))
                # Predicted trajectory segment
                frame_data.append(
                    go.Scatter3d(x=predicted[start_idx:k, i, 0], y=predicted[start_idx:k, i, 1],
                                 z=predicted[start_idx:k, i, 2],
                                 mode='lines', line=dict(width=5, color='red'), opacity=0.3))

                frame_data.append(
                    go.Scatter3d(x=[actual[k, i, 0]], y=[actual[k, i, 1]], z=[actual[k, i, 2]],
                                 mode='markers', marker=dict(size=2, color='blue'), name=f'Actual step {k}'))
                # Marker for the predicted position at step k
                frame_data.append(
                    go.Scatter3d(x=[predicted[k, i, 0]], y=[predicted[k, i, 1]], z=[predicted[k, i, 2]],
                                 mode='markers', marker=dict(size=2, color='red'), name=f'Predicted step {k}'))
            frames.append(go.Frame(data=frame_data, name=f'frame{k}'))

        fig.frames = frames

        # Calculate min and max for all dimensions
        x_min, x_max = np.min(actual[:, :, 0]), np.max(actual[:, :, 0])
        y_min, y_max = np.min(actual[:, :, 1]), np.max(actual[:, :, 1])
        z_min, z_max = np.min(actual[:, :, 2]), np.max(actual[:, :, 2])

        # Adjust for predicted positions
        x_min = min(x_min, np.min(predicted[:, :, 0]))
        x_max = max(x_max, np.max(predicted[:, :, 0]))
        y_min = min(y_min, np.min(predicted[:, :, 1]))
        y_max = max(y_max, np.max(predicted[:, :, 1]))
        z_min = min(z_min, np.min(predicted[:, :, 2]))
        z_max = max(z_max, np.max(predicted[:, :, 2]))

        # Calculate ranges with some padding
        padding = 5  # Adjust padding as necessary
        x_range = [x_min - padding, x_max + padding]
        y_range = [y_min - padding, y_max + padding]
        z_range = [z_min - padding, z_max + padding]

        # Fix the axes ranges and ensure a fixed aspect ratio
        fig.update_layout(scene=dict(
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'),
            title='N-body',
            showlegend=False)

        # Save the plot as an HTML file
        fig.write_html(output_file)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(43)

    sim = GravitySim(n_balls=100, loc_std=1)

    t = time.time()
    loc, vel, force, mass = sim.sample_trajectory(T=5000, sample_freq=1)

    print("Simulation time: {}".format(time.time() - t))

    offset = 4000
    N_frames = loc.shape[0] - offset
    N_particles = loc.shape[-2]

    sim.plot_energies(loc, vel, mass)
