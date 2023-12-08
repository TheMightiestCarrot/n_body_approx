import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import traceback
import os
import glob


def load_model(run, particle_dim, particle_index):
    class ParticlePredictor(nn.Module):
        def __init__(self, particle_dim, model_dim, num_heads, num_layers, particle_index=None):
            super(ParticlePredictor, self).__init__()
            self.particle_index = particle_index
            self.input_linear = nn.Linear(particle_dim, model_dim)
            # encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, activation=leaky_relu)
            encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
            self.output_linear = nn.Linear(model_dim, particle_dim)

        def forward(self, x):
            x = self.input_linear(x)
            x = self.transformer_encoder(x)
            if self.particle_index is not None:
                x = x[:, self.particle_index, :]
            x = self.output_linear(x)
            return x

    parts = run.split('/')[-1].split('_')

    model_dim = int(parts[3])
    num_heads = int(parts[4])
    num_layers = int(parts[5])
    batch_size = int(parts[6])
    lr = float(parts[7])

    model = ParticlePredictor(particle_dim, model_dim, num_heads, num_layers, particle_index)
    # model.load_state_dict(torch.load(os.path.join(run, "encoder_only_pos_vel_d1.pth")))

    models = glob.glob(run + "/" + '*.pth')
    if len(models) > 1:
        print("MORE MODELS FOUND IN THE DIR, LOADING THE FIRST:", models[0])
    model.load_state_dict(torch.load(models[0]))

    return model


def plot_trajectory(actual_data, predicted_data, particle_index, loggers=[], epoch=0, dims=3):
    fig = plt.figure()
    if dims == 3:  # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(actual_data[:, particle_index, 0], actual_data[:, particle_index, 1], actual_data[:, particle_index, 2],
                label='Actual')
        ax.plot(predicted_data[:, particle_index, 0], predicted_data[:, particle_index, 1],
                predicted_data[:, particle_index, 2], label='Predicted')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:  # 2D plot
        ax = fig.add_subplot(111)
        ax.plot(actual_data[:, particle_index, 0], actual_data[:, particle_index, 1], label='Actual')
        ax.plot(predicted_data[:, particle_index, 0], predicted_data[:, particle_index, 1], label='Predicted')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    ax.set_title(f"Trajectory for Particle {particle_index}")
    ax.legend()

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Trajectory/Particle_{particle_index}_epoch_{epoch}'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()


def get_predicted_data(model, inputs_tensor, particle_index):
    model.eval()  # Set the model to evaluation mode
    predicted_data = []
    with torch.no_grad():  # Disable gradient computation
        for i in range(inputs_tensor.shape[0]):
            if particle_index is not None:
                output_tensor = model(inputs_tensor[i].unsqueeze(0)).squeeze(0)  # Remove batch dimension
            else:
                output_tensor = model(inputs_tensor[i]).squeeze(0)  # Remove batch dimension
            predicted_data.append(output_tensor.cpu().numpy())

    predicted_data = np.array(predicted_data)  # Convert list of arrays to a single numpy array
    if particle_index is not None:
        predicted_data = np.expand_dims(predicted_data, axis=1)

    return predicted_data


def calculate_mse(actual_data, predicted_data):
    return ((actual_data - predicted_data) ** 2).mean(axis=1)


def calculate_mae(actual_data, predicted_data):
    return (abs(actual_data - predicted_data)).mean(axis=1)


def calculate_percentage_error(true_data, predicted_data):
    """
    Calculate the percentage error relative to the magnitude of the true data.

    :param true_data: numpy array of true values (either position or velocity).
    :param predicted_data: numpy array of predicted values (either position or velocity).
    :return: numpy array of percentage error values.
    """
    mae = calculate_mae(true_data, predicted_data)
    magnitude = np.linalg.norm(true_data, axis=2)
    magnitude[magnitude == 0] = np.nan  # Avoid division by zero
    return (mae / magnitude) * 100


def plot_error_over_time_position(targets_np, predicted_data, particle_index=None, loggers=[], epoch=0, dims=3):
    if particle_index is not None:
        actual_data = targets_np[:, particle_index, :].reshape(-1, 1, 2 * dims)
    else:
        actual_data = targets_np

    ####################################################################################################################
    # MSE POSITION
    ####################################################################################################################
    # mse_values = calculate_mse(actual_data[:, :, :dims],
    #                            predicted_data[:, :, :dims])  # Considering only position data for MSE
    # time_steps = np.arange(mse_values.shape[0])  # Generate an array of time steps
    # plt.figure()
    # plt.plot(time_steps, mse_values)  # Plot MSE values against time steps
    # plt.xlabel('Time Step')
    # plt.ylabel('Mean Squared Error')
    # plt.title('Position MSE')
    # if writer:
    #     writer.add_figure('Position/MSE_' + str(epoch), plt.gcf(), global_step=epoch)
    # plt.show()

    ####################################################################################################################
    # MAE POSITION
    ####################################################################################################################
    mae_values = calculate_mae(actual_data[:, :, :dims], predicted_data[:, :, :dims])
    time_steps = np.arange(mae_values.shape[0])
    plt.figure()
    plt.plot(time_steps, mae_values)
    plt.xlabel('Time Step')
    plt.ylabel('Mean absolute Error')
    plt.title('Position MAE')

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Position/MAE'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()

    ####################################################################################################################
    # % MAE vs L2 POSITION
    ####################################################################################################################
    # Extract position data
    actual_position = actual_data[:, :, :dims]
    predicted_position = predicted_data[:, :, :dims]

    # Calculate the percentage error for position
    percentage_error = calculate_percentage_error(actual_position, predicted_position)

    # Plotting
    time_steps = np.arange(percentage_error.shape[0])
    plt.figure()
    plt.plot(time_steps, percentage_error)
    plt.xlabel('Time Step')
    plt.ylabel('Positional MAE as % of Position Magnitude')
    plt.title('Positional Error Percentage Relative to Position Magnitude')

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Position/Percentage'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()

    ####################################################################################################################
    # MAE vs L2 VELOCITY
    ####################################################################################################################
    # Extract velocity data
    actual_velocity = actual_data[:, :, dims:]

    # Calculate positional MAE
    positional_mae = calculate_mae(actual_position, predicted_position)

    # Calculate L2 norm of the velocity
    velocity_l2_norm = np.linalg.norm(actual_velocity, axis=2)
    velocity_l2_norm[velocity_l2_norm == 0] = np.nan  # Avoid division by zero

    # Calculate the percentage error
    percentage_error = (positional_mae / velocity_l2_norm) * 100

    # Plotting
    time_steps = np.arange(percentage_error.shape[0])
    plt.figure()
    plt.plot(time_steps, percentage_error)
    plt.xlabel('Time Step')
    plt.ylabel('Positional MAE as % of Velocity L2 Norm')
    plt.title('Positional MAE Percentage Relative to Velocity L2 Norm')

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Position/MAE vs L2 velocity'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()


def plot_error_over_time_velocity(targets_np, predicted_data, particle_index=None, loggers=[], epoch=0, dims=3):
    if particle_index is not None:
        actual_data = targets_np[:, particle_index, :].reshape(-1, 1, 2 * dims)
    else:
        actual_data = targets_np

    # mse_values = calculate_mse(actual_data[:, :, dims:],
    #                            predicted_data[:, :, dims:])  # Considering only position data for MSE
    # time_steps = np.arange(mse_values.shape[0])  # Generate an array of time steps
    # plt.figure()
    # plt.plot(time_steps, mse_values)  # Plot MSE values against time steps
    # plt.xlabel('Time Step')
    # plt.ylabel('Mean Squared Error')
    # plt.title('Velocity MSE')
    # if writer:
    #     writer.add_figure('Velocity/MSE_' + str(epoch), plt.gcf(), global_step=epoch)
    # plt.show()

    mae_values = calculate_mae(actual_data[:, :, dims:], predicted_data[:, :, dims:])
    time_steps = np.arange(mae_values.shape[0])
    plt.figure()
    plt.plot(time_steps, mae_values)
    plt.xlabel('Time Step')
    plt.ylabel('Mean absolute Error')
    plt.title('Velocity MAE')

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Velocity/MAE'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()

    ####################################################################################################################
    # % MAE vs L2 VELOCITY
    ####################################################################################################################
    # Extract velocity data
    actual_velocity = actual_data[:, :, dims:]
    predicted_velocity = predicted_data[:, :, dims:]

    # Calculate the percentage error for velocity
    percentage_error = calculate_percentage_error(actual_velocity, predicted_velocity)

    # Plotting
    time_steps = np.arange(percentage_error.shape[0])
    plt.figure()
    plt.plot(time_steps, percentage_error)
    plt.xlabel('Time Step')
    plt.ylabel('Velocity MAE as % of Velocity Magnitude')
    plt.title('Velocity Error Percentage Relative to Velocity Magnitude')

    if loggers:
        fig = plt.gcf()  # Your matplotlib figure
        tag = f'Velocity/Percentage'
        for logger in loggers:
            logger.log_figure(tag, fig, epoch)

    plt.show()


def interactive_trajectory_plot(actual_data, predicted_data, particle_index=None, boxSize=1, dims=3, offline_plot=False):
    import matplotlib
    og_backend = matplotlib.get_backend()
    try:
        if offline_plot:
            matplotlib.use('Agg')
        else:
            try:
                matplotlib.use('Qt5Agg')
            except (NameError, KeyError):
                matplotlib.use('TkAgg')
        plt.ion()
        fig = plt.figure(figsize=(12, 8))

        if dims == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-boxSize, boxSize)
            ax.set_ylim(-boxSize, boxSize)
            ax.set_zlim(-boxSize, boxSize)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax = fig.add_subplot(111)
            ax.set_xlim(-boxSize, boxSize)
            ax.set_ylim(-boxSize, boxSize)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        plt.subplots_adjust(bottom=0.25)
        actual_line, = ax.plot([], [], 'b-', label='Actual')
        predicted_line, = ax.plot([], [], 'r-', label='Predicted')
        ax.legend()
        ax.set_title(
            'Trajectory of predicted particle' + ('s' if particle_index is None else f' {str(particle_index)}'))
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, 'Time Step', 0, actual_data.shape[0] - 1, valinit=0, valfmt='%0.0f')

        def update(val):
            time_step = int(slider.val)
            window_size = 50
            start_step = max(0, time_step - window_size)

            if dims == 3:
                actual_line.set_data(actual_data[start_step:time_step + 1, particle_index, 0],
                                     actual_data[start_step:time_step + 1, particle_index, 1])
                actual_line.set_3d_properties(actual_data[start_step:time_step + 1, particle_index, 2])

                predicted_line.set_data(predicted_data[start_step:time_step + 1, particle_index, 0],
                                        predicted_data[start_step:time_step + 1, particle_index, 1])
                predicted_line.set_3d_properties(predicted_data[start_step:time_step + 1, particle_index, 2])


            else:
                actual_line.set_data(actual_data[start_step:time_step + 1, particle_index, 0],
                                     actual_data[start_step:time_step + 1, particle_index, 1])
                predicted_line.set_data(predicted_data[start_step:time_step + 1, particle_index, 0],
                                        predicted_data[start_step:time_step + 1, particle_index, 1])

            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == 'left':
                new_val = max(0, slider.val - 1)
                slider.set_val(new_val)
            elif event.key == 'right':
                new_val = min(actual_data.shape[0] - 1, slider.val + 1)
                slider.set_val(new_val)

        fig.canvas.mpl_connect('key_press_event', on_key)

        update(0)
        slider.on_changed(update)
        plt.show(block=True)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        traceback.print_exc()

    matplotlib.use(og_backend)


def interactive_trajectory_plot_all_particles(actual_data, predicted_data, particle_index=None, boxSize=1, dims=3):
    plt.ion()
    fig = plt.figure(figsize=(12, 8))

    if dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
        ax.set_zlim(-boxSize, boxSize)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    ax.set_title('Trajectories of all particles ' + (
        '' if particle_index is None else f'with predicted particle {str(particle_index)}'))

    plt.subplots_adjust(bottom=0.25)
    actual_lines = [ax.plot([], [], 'b-', label='Actual')[0] for _ in range(actual_data.shape[1])]
    predicted_line, = ax.plot([], [], 'r-', label='Predicted')
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color='blue', label='Actual'), predicted_line]
    ax.legend(handles=legend_lines)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, actual_data.shape[0] - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        time_step = int(slider.val)
        window_size = 50
        start_step = max(0, time_step - window_size)

        if dims == 2:
            for pi in range(actual_data.shape[1]):
                actual_lines[pi].set_data(actual_data[start_step:time_step + 1, pi, 0],
                                          actual_data[start_step:time_step + 1, pi, 1])

                if pi == particle_index:
                    predicted_line.set_data(predicted_data[start_step:time_step + 1, particle_index, 0],
                                            predicted_data[start_step:time_step + 1, particle_index, 1])
        if dims == 3:
            for pi in range(actual_data.shape[1]):
                actual_lines[pi].set_data(actual_data[start_step:time_step + 1, pi, 0],
                                          actual_data[start_step:time_step + 1, pi, 1])
                actual_lines[pi].set_3d_properties(actual_data[start_step:time_step + 1, pi, 2])

                if pi == particle_index:
                    predicted_line.set_data(predicted_data[start_step:time_step + 1, particle_index, 0],
                                            predicted_data[start_step:time_step + 1, particle_index, 1])
                    predicted_line.set_3d_properties(predicted_data[start_step:time_step + 1, particle_index, 2])

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'left':
            new_val = max(0, slider.val - 1)
            slider.set_val(new_val)
        elif event.key == 'right':
            new_val = min(actual_data.shape[0] - 1, slider.val + 1)
            slider.set_val(new_val)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update(0)
    slider.on_changed(update)
    plt.show(block=True)


def interactive_trajectory_plot_just_sim(actual_data, particle_index=None, boxSize=1, dims=3):
    plt.ion()
    fig = plt.figure(figsize=(12, 8))

    # Conditionally set up the subplot for 2D or 3D plotting
    if dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(-boxSize, boxSize)
        ax.set_zlabel('Z')
    elif dims == 2:
        ax = fig.add_subplot(111)
    else:
        raise ValueError("dims must be 2 or 3")

    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim(-boxSize, boxSize)
    ax.set_ylim(-boxSize, boxSize)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, actual_data.shape[0] - 1, valinit=0, valfmt='%0.0f')

    # Update function for the slider
    def update(val):
        time_step = int(slider.val)
        window_size = 10
        start_step = max(0, time_step - window_size)
        ax.clear()
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
        if dims == 3:
            ax.set_zlim(-boxSize, boxSize)

        for pi in range(actual_data.shape[1]):
            if dims == 3:
                ax.plot(actual_data[start_step:time_step + 1, pi, 0],
                        actual_data[start_step:time_step + 1, pi, 1],
                        actual_data[start_step:time_step + 1, pi, 2], 'b-')
            elif dims == 2:
                ax.plot(actual_data[start_step:time_step + 1, pi, 0],
                        actual_data[start_step:time_step + 1, pi, 1], 'b-')

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'left':
            new_val = max(0, slider.val - 1)
            slider.set_val(new_val)
        elif event.key == 'right':
            new_val = min(actual_data.shape[0] - 1, slider.val + 1)
            slider.set_val(new_val)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update(0)
    slider.on_changed(update)
    plt.show(block=True)


def self_feed(model, inputs_tensor, particle_index=None, number_of_predictions=2000, reset_every=120):
    model.eval()  # Set the model to evaluation mode
    predicted_data_self_feed = []
    first_step = 0

    with torch.no_grad():  # Disable gradient computation
        current_input = inputs_tensor[first_step, :, :].clone().detach()
        for i in range(number_of_predictions):

            if particle_index is not None:
                output_tensor = model(current_input.unsqueeze(0)).squeeze(0)
            else:
                output_tensor = model(current_input.unsqueeze(0))
            predicted_data_self_feed.append(output_tensor.cpu().numpy())

            current_input = inputs_tensor[i + 1, :, :].clone().detach()
            if i % reset_every != 0:
                current_input[particle_index] = output_tensor

    predicted_data_self_feed = np.array(predicted_data_self_feed)  # Convert list of arrays to a single numpy array
    if particle_index is not None:
        predicted_data_self_feed = np.expand_dims(predicted_data_self_feed, axis=1)
    return predicted_data_self_feed


def simulate_gravitational_system(seed_value, N, tEnd, dt, softening, G, boxSize, mass_coef, dims=3, init_boxsize=None):
    t = 0

    def getAcc(pos, mass, G, softening):
        # Determine the number of dimensions from the position array
        num_dimensions = pos.shape[1]

        # Initialize an empty list to store acceleration components
        acc_components = []

        # Loop over each dimension to calculate pairwise separations and accelerations
        for dim in range(num_dimensions):
            # Get the positions for this dimension
            r_dim = pos[:, dim:dim + 1]

            # Calculate pairwise separations for this dimension
            dr_dim = r_dim.T - r_dim

            # Append to the list of acceleration components
            acc_components.append(dr_dim)

        # Calculate the matrix for 1/r^3 for all particle pairwise separations
        inv_r3 = softening ** 2
        for dr_dim in acc_components:
            inv_r3 += dr_dim ** 2
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

        # Calculate acceleration for each dimension
        a = []
        for dr_dim in acc_components:
            a_dim = G * (dr_dim * inv_r3) @ mass
            a.append(a_dim)

        # Combine acceleration components for all dimensions
        a = np.hstack(a)

        return a

    def getEnergy(pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * np.sum(np.sum(mass * vel ** 2))

        num_particles, num_dimensions = pos.shape
        inv_r = np.zeros((num_particles, num_particles))

        for dim in range(num_dimensions):
            dim_diff = pos[:, dim:dim + 1].T - pos[:, dim:dim + 1]
            inv_r += dim_diff ** 2

        inv_r = np.sqrt(inv_r)
        inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(mass * mass.T) * inv_r, 1)))

        return KE, PE

    # Generate Initial Conditions
    np.random.seed(seed_value)  # set the random number generator seed

    # mass = mass_coef*np.ones((N,1))/N  # total mass of particles is 20
    # pos = np.random.rand(N, 3) * 2 * boxSize - boxSize
    # vel  = np.random.randn(N,3)

    if init_boxsize is None:
        init_boxsize = boxSize

    pos = np.random.rand(N, dims).astype(np.float64) * 2 * init_boxsize - init_boxsize
    vel = np.random.randn(N, dims).astype(np.float64) * 0.8
    mass = (mass_coef * np.ones((N, 1)) / N).astype(np.float64)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    pos_save = np.zeros((N, dims, Nt + 1))
    vel_save = np.zeros((N, dims, Nt + 1))
    pos_save[:, :, 0] = pos
    vel_save[:, :, 0] = vel
    KE_save = np.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = np.arange(Nt + 1) * dt

    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt

        for n in range(N):
            for j in range(dims):
                if pos[n, j] > boxSize:
                    pos[n, j] = boxSize - (pos[n, j] - boxSize)  # Reflect position inside boundary
                    vel[n, j] *= -1  # Reverse velocity component
                elif pos[n, j] < -boxSize:
                    pos[n, j] = -boxSize + (-boxSize - pos[n, j])  # Reflect position inside boundary
                    vel[n, j] *= -1  # Reverse velocity component

        acc = getAcc(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        pos_save[:, :, i + 1] = pos
        vel_save[:, :, i + 1] = vel
        # get energy of system
        KE, PE = getEnergy(pos, vel, mass, G)
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

    combined_data = np.concatenate((pos_save, vel_save), axis=1)  # Shape: (100, 6, 1001)
    combined_data = combined_data.transpose(2, 0, 1)

    filter_condition = ((KE_save + PE_save) > -200)
    combined_data = combined_data[filter_condition, :, :]
    filtered_KE = KE_save[filter_condition]
    filtered_PE = PE_save[filter_condition]
    filtered_total_energy = filtered_KE + filtered_PE
    filtered_time = t_all[filter_condition]

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_time, filtered_KE, label='Kinetic Energy', color='red')
    plt.plot(filtered_time, filtered_PE, label='Potential Energy', color='blue')
    plt.plot(filtered_time, filtered_total_energy, label='Total Energy', color='black')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy vs Time (Filtered)')
    plt.legend()
    plt.grid(True)
    plt.show()

    if dims == 2:
        plt.figure(figsize=(10, 8))
        for n in range(N):
            plt.plot(pos_save[n, 0, :], pos_save[n, 1, :], label=f'Particle {n + 1}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
    elif dims == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for n in range(N):
            ax.plot(pos_save[n, 0, :], pos_save[n, 1, :], pos_save[n, 2, :], label=f'Particle {n + 1}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
    else:
        raise ValueError("Dimensions not supported for plotting")

    plt.title('Particle Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

    return combined_data


def log_hparams(writer, hparams, loss):
    writer.add_hparams(hparams, {'train_loss': loss})
    for key, value in hparams.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value)


def plot_hist_of_simulation_data(combined_data, dims=3):
    data_for_dislpay_dist = combined_data[:, :, [0, dims]]

    # Reshape the array to flatten the first two dimensions
    flattened_data = data_for_dislpay_dist.reshape(-1, data_for_dislpay_dist.shape[2])

    # Generate histograms for each feature
    for i in range(flattened_data.shape[1]):
        feature_data = flattened_data[:, i]
        histogram, bin_edges = np.histogram(feature_data, bins=50)

        plt.figure()
        plt.hist(feature_data, bins=bin_edges, alpha=0.75, color='blue')
        plt.title(f'Histogram of Feature {i + 1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()


def process_data(combined_data, dims=3):
    position_data = combined_data[:, :, :dims]
    velocity_data = combined_data[:, :, dims:]

    # Reshape the data
    reshaped_position_data = position_data.reshape(-1, position_data.shape[2])
    reshaped_velocity_data = velocity_data.reshape(-1, velocity_data.shape[2])

    # Find global min and max for position data
    position_min = reshaped_position_data.min()
    position_max = reshaped_position_data.max()

    # Find global min and max for velocity data
    velocity_min = reshaped_velocity_data.min()
    velocity_max = reshaped_velocity_data.max()

    # Apply min-max scaling
    scaled_position_data = -1 + 2 * (reshaped_position_data - position_min) / (position_max - position_min)
    scaled_velocity_data = -1 + 2 * (reshaped_velocity_data - velocity_min) / (velocity_max - velocity_min)

    # Reshape scaled data back to original format, if needed
    scaled_position_data = scaled_position_data.reshape(position_data.shape)
    scaled_velocity_data = scaled_velocity_data.reshape(velocity_data.shape)

    # Combine scaled data back into original format, if needed
    scaled_combined_data = np.concatenate([scaled_position_data, scaled_velocity_data], axis=2)

    inputs = []
    targets = []
    # skip first 20 stepov pre istotu nech sa to utrasie
    skip_first = 0
    for i in range(skip_first, scaled_combined_data.shape[0] - 1):
        inputs.append(scaled_combined_data[i, :, :])
        targets.append(scaled_combined_data[i + 1, :, :])

    inputs_np = np.array(inputs)  # Shape: [1000, 100, 6]
    targets_np = np.array(targets)  # Shape: [1000, 100, 6]

    print("combined_data shape:", scaled_combined_data.shape)
    print("inputs_np shape:", inputs_np.shape)
    print("targets_np shape:", targets_np.shape)

    plot_hist_of_simulation_data(combined_data, dims=dims)
    print("Normalized:")
    plot_hist_of_simulation_data(scaled_combined_data, dims=dims)

    return inputs_np, targets_np


def train_model(model, optimizer, criterion, data_loader, num_epochs, old_epoch=0, loggers=[], dims=3):
    try:
        print("training")
        last_avg_loss = 0
        model.train()
        for epoch in range(old_epoch + 1, old_epoch + num_epochs):
            total_metrics = {
                "loss": 0, "loss_pos": 0, "loss_vel": 0,
                "perc_error_pos": 0, "perc_error_vel": 0, "perc_error_pos_vs_vel_l1": 0, "perc_error_pos_vs_vel_l2": 0
            }
            num_batches = 0
            for batch in data_loader:
                inputs, targets = batch  # Adjust based on your data loading method

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predicted_pos = outputs[..., :dims]
                    target_pos = targets[..., :dims]

                    predicted_vel = outputs[..., dims:]
                    target_vel = targets[..., dims:]

                    loss_pos = criterion(predicted_pos, target_pos)
                    loss_vel = criterion(predicted_vel, target_vel)

                    # Calculate percentage errors
                    perc_error_pos = (torch.norm(predicted_pos - target_pos, dim=1) /
                                      torch.norm(target_pos, dim=1)).mean() * 100

                    perc_error_vel = (torch.norm(predicted_vel - target_vel, dim=1) /
                                      torch.norm(target_vel, dim=1)).mean() * 100

                    perc_error_pos_vs_vel_l1 = (torch.abs(predicted_pos - target_pos).mean() /
                                                torch.norm(target_vel, dim=1)).mean() * 100

                    perc_error_pos_vs_vel_l2 = (torch.norm(predicted_pos - target_pos, dim=1) /
                                                torch.norm(target_vel, dim=1)).mean() * 100

                    total_metrics["loss"] += loss.item()
                    total_metrics["loss_pos"] += loss_pos.item()
                    total_metrics["loss_vel"] += loss_vel.item()
                    total_metrics["perc_error_pos"] += perc_error_pos.item()
                    total_metrics["perc_error_vel"] += perc_error_vel.item()
                    total_metrics["perc_error_pos_vs_vel_l1"] += perc_error_pos_vs_vel_l1.item()
                    total_metrics["perc_error_pos_vs_vel_l2"] += perc_error_pos_vs_vel_l2.item()

                num_batches += 1

            if epoch % 1 == 0:
                avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], avg_both: {avg_metrics['loss']:.5f}, avg_pos: {avg_metrics['loss_pos']: .5f}, avg_vel: {avg_metrics['loss_vel']: .5f}, perc_pos: {avg_metrics['perc_error_pos']: .5f}%, perc_vel: {avg_metrics['perc_error_vel']: .5f}%")

                for logger in loggers:
                    # writer.add_scalar('Loss/last_both', total_metrics["loss"], epoch)
                    logger.log_scalar('Loss/last_pos', loss_pos.item(), epoch)
                    logger.log_scalar('Loss/last_vel', loss_vel.item(), epoch)

                    # logger.log_scalar('Loss/avg_both', avg_metrics["loss"], epoch)
                    logger.log_scalar('Loss/avg_pos', avg_metrics["loss_pos"], epoch)
                    logger.log_scalar('Loss/avg_vel', avg_metrics["loss_vel"], epoch)

                    logger.log_scalar('Loss/perc_pos', avg_metrics["perc_error_pos"], epoch)
                    logger.log_scalar('Loss/perc_vel', avg_metrics["perc_error_vel"], epoch)
                    logger.log_scalar('Loss/perc_pos_vs_vel_l1', avg_metrics["perc_error_pos_vs_vel_l1"], epoch)
                    logger.log_scalar('Loss/perc_pos_vs_vel_l2', avg_metrics["perc_error_pos_vs_vel_l2"], epoch)

                    for name, weight in model.named_parameters():
                        logger.log_histogram(f'{name}/weights', weight, epoch)
                        if weight.grad is not None:
                            logger.log_histogram(f'{name}/grads', weight.grad, epoch)

                    last_avg_loss = avg_metrics["loss"]


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()

    print("Saving model at the end of training")
    for logger in loggers:
        if logger.get_logdir():
            torch.save(model, os.path.join(logger.get_logdir(), "model.pth"))

    return epoch, last_avg_loss


# TODO 3d dims
def interactive_trajectory_plot_all_particles(actual_data, predicted_data, particle_index=None, boxSize=1, dims=3,
                                              offline_plot=False, loggers=[], video_tag="trajectories all particles"):
    import matplotlib
    og_backend = matplotlib.get_backend()
    try:
        if offline_plot:
            matplotlib.use('Agg')
        else:
            try:
                matplotlib.use('Qt5Agg')
            except (NameError, KeyError):
                matplotlib.use('TkAgg')

        skip_steps = 2
        trace_length = 10
        plt.ion()
        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(111)
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_title('Trajectories of all particles ' + (
            '' if particle_index is None else f'with predicted particle {str(particle_index)}'))

        plt.subplots_adjust(bottom=0.25)
        actual_lines = [ax.plot([], [], 'b-', label='Actual')[0] for _ in range(actual_data.shape[1])]
        predicted_line, = ax.plot([], [], 'r-', label='Predicted')
        from matplotlib.lines import Line2D
        legend_lines = [Line2D([0], [0], color='blue', label='Actual'), predicted_line]
        ax.legend(handles=legend_lines)

        if not offline_plot:
            ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider = Slider(ax_slider, 'Time Step', 0, actual_data.shape[0] - 1, valinit=0, valfmt='%0.0f')

        def update(val):
            if offline_plot:
                time_step = int(val) * skip_steps
            else:
                time_step = int(val)

            start_step = max(0, time_step - trace_length)

            for pi in range(actual_data.shape[1]):
                actual_lines[pi].set_data(actual_data[start_step:time_step + 1, pi, 0],
                                          actual_data[start_step:time_step + 1, pi, 1])

                if pi == particle_index:
                    predicted_line.set_data(predicted_data[start_step:time_step + 1, particle_index, 0],
                                            predicted_data[start_step:time_step + 1, particle_index, 1])

            if offline_plot:
                return actual_lines + [predicted_line]
            else:
                fig.canvas.draw_idle()

        def on_key(event):
            if event.key == 'left':
                new_val = max(0, slider.val - 1)
                slider.set_val(new_val)
            elif event.key == 'right':
                new_val = min(actual_data.shape[0] - 1, slider.val + 1)
                slider.set_val(new_val)

        if offline_plot:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp:
                from matplotlib.animation import FuncAnimation
                fps = 20
                frames = actual_data.shape[0] // skip_steps
                anim = FuncAnimation(fig, update, frames=frames, blit=True)
                filename = temp.name
                anim.save(filename, fps=fps,
                          extra_args=['-vcodec', 'libx264', '-preset', 'fast', '-crf', '22'])

                for i, logger in enumerate(loggers):
                    logger.log_video(f"{video_tag}", filename, fps=fps)

        else:
            fig.canvas.mpl_connect('key_press_event', on_key)

            update(0)
            slider.on_changed(update)
            plt.show(block=True)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        traceback.print_exc()

    matplotlib.use(og_backend)
