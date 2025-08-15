import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tf import transformations
import datetime
from matplotlib.ticker import FixedLocator


from deps.utils.util import keep_max_n_files

def seconds_to_timestamp(seconds):
    time_correction = 0
    # time_correction = 3600*1 + 3*60 + 38
    seconds -= time_correction
    # render as HH:MM:SS
    return datetime.datetime.fromtimestamp(seconds).strftime('%H:%M:%S')

def euler_from_quaternion(q):
    """Convert quaternion to euler angles"""
    q = np.array(q).reshape(-1, 4)
    euler = []
    for i in range(len(q)):
        euler.append(transformations.euler_from_quaternion(q[i]))
    return np.asarray(euler)*180/np.pi
    
def draw_occlusions(ax, occlusions_dict):
    # create red overlay for occlusions
    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    for i, (frame, occlusions) in enumerate(occlusions_dict.items()):
        try:
            i = int(''.join(filter(str.isdigit, frame)))-1
        except ValueError:
            i = i
        color = colors[i % len(colors)]
        for occlusion in occlusions:
            ax.axvspan(occlusion[0], occlusion[1], color=color, alpha=0.1)
            # write camera name
            y_pos = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.3
            ax.text(occlusion[0], y_pos, frame, rotation=90, fontsize=8, color=color)


def save_to_file(data, fusion_name, save_dir, occlusions, t_bag_start=None, show_position=False, show_orientation=False, show_3d=False):
    """Input is a dict containing the input and output in IMU frame"""
    print(f'Saving results to {save_dir}') if save_dir else None
    # convert list to np array for easier operation
    data_np = {}
    for meas_type, samples in data.items():
        if samples:
            print(f'Found data for {meas_type}')
            data_np[meas_type] = np.asarray(samples)
    if not data_np:
        print("Nothing to plot..")
        return 
    
    use_ots = True
    use_isam2 = False

    use_imu_rt = False
    use_imu_isam2 = True

    use_hh_mm_ss = True

    show_axes = [1,1,1]

    # save it to file as '#time(ns)', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz'
    # first create array that contains all data
    # save to file
    p, q, v, ba, bw = None, None, None, None, None
    for meas_type, samples in data_np.items():
        # if meas_type == 'IMU_ISAM2':
        #     p = samples[:, 1:4]
        if meas_type == fusion_name:
            # samples is organized as follows:
            t = samples[:, 0]
            p = samples[:, 1:4]
            q = samples[:, 4:8]
            v = samples[:, 8:11]
            ba = samples[:, 11:14]
            bw = samples[:, 14:17]
            # convert time to ns
            t = t * 1e9
            # convert quaternion to wxyz
            q = np.hstack((q[:, 3].reshape(-1, 1), q[:, 0:3]))

        if p is not None and q is not None and v is not None and ba is not None and bw is not None:  
            # pad p with nan if it is shorter than t
            if len(p) < len(t):
                p = np.pad(p, ((0, len(t) - len(p)), (0, 0)), mode='constant', constant_values=np.nan)

            output = np.hstack((t.reshape(-1, 1), p, q, v, bw, ba))
            df = pd.DataFrame(output, columns=['#time(ns)', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz'])
            df.to_csv(os.path.join(save_dir, 'output.csv'), index=False)

    # # save occlusions as interrupts.txt
    # if occlusions is not None:
    #     with open(os.path.join(save_dir, 'interrupts.txt'), 'w') as f:
    #         for occlusion in occlusions:
    #             f.write(f'{occlusion[0]}\t{occlusion[1]}\n')

    t_imu = []
    acc = []
    gyro = []
    for meas_type, samples in data_np.items():
        if meas_type == 'IMU':
            # samples is organized as follows:
            # entry 0 is time, 1 is dt, 2-4 is acc, 5-7 is gyro
            t_imu.append(samples[:, 0])
            acc.append(samples[:, 2:5])
            gyro.append(samples[:, 5:8])

    # plot accelerometer
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.scatter(t_imu, acc[0][:, 0], s=1) if show_axes[0] else None
    # ax.scatter(t_imu, acc[0][:, 1], s=1) if show_axes[1] else None
    # ax.scatter(t_imu, acc[0][:, 2], s=1) if show_axes[2] else None
    for i in range(3):
        ax.scatter(t_imu, acc[0][:, i], s=1) if show_axes[i] else None
        # ax.plot(np.array(t_imu).reshape(-1), acc[0][:, i], label=f'{i}', linewidth=1) if show_axes[i] else None
        # plot diff (make sure t and acc have same length)
        # ax.scatter(np.array(t_imu).reshape(-1)[1:], np.abs(np.diff(acc[0][:, i])), s=1) if show_axes[i] else None

    # parse time to be timestamps
    if use_hh_mm_ss:
        # tick_positions = ax.get_xticks()
        # create ticks every 10 seconds
        tick_positions = np.arange(t_imu[0][0], t_imu[0][-1], 10)
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.set_xticklabels([seconds_to_timestamp(t) for t in tick_positions])
        # 45 degree rotation
        plt.xticks(rotation=45)
        
    ax.set_title('Accelerometer')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.legend(['x', 'y', 'z'])
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accelerometer.png'), dpi=300)
    plt.close()

    # plot gyro
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(t_imu, gyro[0][:, 0], s=1) if show_axes[0] else None
    ax.scatter(t_imu, gyro[0][:, 1], s=1) if show_axes[1] else None
    ax.scatter(t_imu, gyro[0][:, 2], s=1) if show_axes[2] else None
    ax.set_title('Gyro')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.legend(['x', 'y', 'z'])
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gyro.png'), dpi=300)
    plt.close()

            
    t_ots = []
    pos_ots = []
    euler_ots = []
    for meas_type, samples in data_np.items():
        if meas_type == 'OTS':
            # samples is organized as follows:
            t_ots.append(samples[:, 0])
            pos_ots.append(samples[:, 1:4])
            euler_ots.append(samples[:, 4:7])

    frames_ots = data_np['OTS_frame']
    frames_ots_unique = np.unique(frames_ots)
    # sort frames_ots_unique
    frames_ots_unique = sorted(frames_ots_unique)
    # sort it such that _inactive come at end
    frames_ots_unique_inactive = [frame for frame in frames_ots_unique if '_inactive' in frame]
    frames_ots_unique = [frame for frame in frames_ots_unique if '_inactive' not in frame]
    frames_ots_unique.extend(frames_ots_unique_inactive)
    print(f'frames_ots_unique: {frames_ots_unique}')

    t_isam = []
    pos_isam = []
    q_isam = []
    ba_isam = []
    bw_isam = []
    for meas_type, samples in data_np.items():
        if meas_type == 'ISAM2':
            # samples is organized as follows:
            t_isam.append(samples[:, 0])
            pos_isam.append(samples[:, 1:4])
            q_isam.append(samples[:, 4:8])
            ba_isam.append(samples[:, 8:11])
            bw_isam.append(samples[:, 11:14])

    if use_imu_isam2:
        t_imu_isam = []
        pos_imu_isam = []
        q_imu_isam = []
        for meas_type, samples in data_np.items():
            if meas_type == 'IMU_ISAM2':
                # samples is organized as follows:
                t_imu_isam.append(samples[:, 0])
                pos_imu_isam.append(samples[:, 1:4])
                q_imu_isam.append(samples[:, 4:8])

    # plot accelerometer bias
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    t = data_np[fusion_name][:, 0]
    ax.scatter(t, ba[:, 0], s=1) 
    ax.scatter(t, ba[:, 1], s=1)
    ax.scatter(t, ba[:, 2], s=1)

    # if use_isam2:
    # plot accelerometer bias of ISAM as black dots
    t_isam = np.vstack(t_isam)
    ba_isam = np.vstack(ba_isam)
    print(f'acc_isam shape: {ba_isam.shape}')
    ax.scatter(t_isam, ba_isam[:, 0], s=1, c='k')
    ax.scatter(t_isam, ba_isam[:, 1], s=1, c='k')
    ax.scatter(t_isam, ba_isam[:, 2], s=1, c='k')


    ax.set_title('Accelerometer bias')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (m/s^2)')
    ax.legend(['x', 'y', 'z'])
    ax.grid()
    plt.tight_layout()
    keep_max_n_files(os.path.join(save_dir, 'accelerometer_bias.png'))
    plt.savefig(os.path.join(save_dir, 'accelerometer_bias.png'), dpi=300)
    plt.close()

    # colors are light and dark of same color (dark green, light green, dark blue, light blue, dark red, light red)
    colors = ['#006400', '#32CD32', '#00008B', '#0000FF', '#8B0000', '#FF0000']
    s_pos = 1 if show_position else 1
    s_ori = 1 if show_orientation else 1

    # plot position
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.grid()

    # plot ots position as black dots
    pos_ots = np.vstack(pos_ots)
    if not show_position:
        init_pos = pos_ots[0]
        # add small offsets make distinguishable on plot in case of rest
        init_pos[1] += 0.005
        init_pos[2] -= 0.005
    else:
        init_pos = np.zeros(3)

    p_filt = np.array(pos_ots - init_pos)
    # keep only selected axes
    p_filt = p_filt[:, np.where(show_axes)[0]]
    # get bounds for p
    p_bounds = [np.min(p_filt), np.max(p_filt)]
    # add 10% margin
    p_range = p_bounds[1] - p_bounds[0]
    p_bounds[0] -= 0.1 * p_range
    p_bounds[1] += 0.1 * p_range
    cm_round = 1
    fac = 100 / cm_round
    p_bounds[0] = np.floor(p_bounds[0] * fac) / fac
    p_bounds[1] = np.ceil(p_bounds[1] * fac) / fac
    print(f'{p_bounds=}')
    # same for t
    t_bounds = [np.min(t_ots), np.max(t_ots)]
    # 1 sec
    t_range = t_bounds[1] - t_bounds[0]
    t_bounds[0] -= 0.1 * t_range
    t_bounds[1] += 0.1 * t_range
    t_bounds[0] = np.floor(t_bounds[0] * 10) / 10
    t_bounds[1] = np.ceil(t_bounds[1] * 10) / 10
    print(f'{t_bounds=}')

    ax.set_xlim(t_bounds[0], t_bounds[1])
    ax.set_ylim(p_bounds[0], p_bounds[1])
    draw_occlusions(ax, occlusions)

    if use_ots:
        t_ots = np.array(t_ots).reshape(-1)
        pos_ots = pos_ots - init_pos
        s_ots = 20
        m_ots_list = ['+', 'x', 's', 'd', 'p', 'h', 'H', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        c_ots_list = ['goldenrod', 'violet', 'darkblue', 'darkred', 'darkcyan', 'darkmagenta']
        for i, frame in enumerate(frames_ots_unique):
            c_ots = c_ots_list[i]
            if '_inactive' in frame:
                try:
                    i = frames_ots_unique.index(frame.replace('_inactive', ''))
                except ValueError:
                    pass
                c_ots = 'gray'
            m_ots = m_ots_list[i]
            idx = np.where(frames_ots == frame)[0]
            ax.scatter(t_ots[idx], pos_ots[idx, 0], s=s_ots, c=c_ots, marker=m_ots) if show_axes[0] else None
            ax.scatter(t_ots[idx], pos_ots[idx, 1], s=s_ots, c=c_ots, marker=m_ots) if show_axes[1] else None
            ax.scatter(t_ots[idx], pos_ots[idx, 2], s=s_ots, c=c_ots, marker=m_ots) if show_axes[2] else None

        # lw = 1
        # ax.plot(t_ots.reshape(-1), pos_ots[:, 0], c='k', label='x (OTS)', linestyle='--', linewidth=lw) if show_axes[0] else None
        # ax.plot(t_ots.reshape(-1), pos_ots[:, 1], c='k', label='y (OTS)', linestyle='--', linewidth=lw) if show_axes[1] else None 
        # ax.plot(t_ots.reshape(-1), pos_ots[:, 2], c='k', label='z (OTS)', linestyle='--', linewidth=lw) if show_axes[2] else None



    if use_isam2:
        # plot isam position as red dots
        pos_isam = np.vstack(pos_isam)
        pos_isam = pos_isam - init_pos
        t_isam = np.vstack(t_isam)
        ax.scatter(t_isam, pos_isam[:, 0], s=4, c='r') if show_axes[0] else None
        ax.scatter(t_isam, pos_isam[:, 1], s=4, c='r') if show_axes[1] else None
        ax.scatter(t_isam, pos_isam[:, 2], s=4, c='r') if show_axes[2] else None
        # lw = 1
        # ax.plot(t_isam.reshape(-1), pos_isam[:, 0], c='r', label='x (ISAM2)', linestyle='--', linewidth=lw) if show_axes[0] else None
        # ax.plot(t_isam.reshape(-1), pos_isam[:, 1], c='r', label='y (ISAM2)', linestyle='--', linewidth=lw) if show_axes[1] else None
        # ax.plot(t_isam.reshape(-1), pos_isam[:, 2], c='r', label='z (ISAM2)', linestyle='--', linewidth=lw) if show_axes[2] else None
    
    # subtract the initial position
    if use_imu_rt:
        p = p - init_pos
        ax.scatter(t, p[:, 0], s=s_pos, label='x', c=colors[1]) if show_axes[0] else None
        ax.scatter(t, p[:, 1], s=s_pos, label='y', c=colors[3]) if show_axes[1] else None
        ax.scatter(t, p[:, 2], s=s_pos, label='z', c=colors[5]) if show_axes[2] else None

    # plot imu_isam position as green dots
    if use_imu_isam2:
        pos_imu_isam = np.vstack(pos_imu_isam)
        pos_imu_isam = pos_imu_isam - init_pos  
        t_imu_isam = np.vstack(t_imu_isam)
        ax.scatter(t_imu_isam, pos_imu_isam[:, 0], s=s_pos, c=colors[4], label='x (IMU_ISAM2)') if show_axes[0] else None
        ax.scatter(t_imu_isam, pos_imu_isam[:, 1], s=s_pos, c=colors[0], label='y (IMU_ISAM2)') if show_axes[1] else None 
        ax.scatter(t_imu_isam, pos_imu_isam[:, 2], s=s_pos, c=colors[2], label='z (IMU_ISAM2)') if show_axes[2] else None
        # # sort t_imu_isam by time
        # t_imu_isam = t_imu_isam.reshape(-1)
        # idx = np.argsort(t_imu_isam)
        # t_imu_isam = t_imu_isam[idx]
        # pos_imu_isam = pos_imu_isam[idx]
        # ax.plot(t_imu_isam, pos_imu_isam[:, 0], c=colors[4], label='x (IMU_ISAM2)', linewidth=1) if show_axes[0] else None
        # ax.plot(t_imu_isam, pos_imu_isam[:, 1], c=colors[0], label='y (IMU_ISAM2)', linewidth=1) if show_axes[1] else None
        # ax.plot(t_imu_isam, pos_imu_isam[:, 2], c=colors[2], label='z (IMU_ISAM2)', linewidth=1) if show_axes[2] else None

    if use_hh_mm_ss:
        # tick_positions = ax.get_xticks()
        # create ticks every 10 seconds
        tick_positions = np.arange(t_bounds[0], t_bounds[1], 10)
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.set_xticklabels([seconds_to_timestamp(t) for t in tick_positions])
        # 45 degree rotation
        plt.xticks(rotation=45)

    title = f'Position (initial position: {np.round(init_pos, 2)})'
    if t_bag_start is not None:
        title += f', start at {seconds_to_timestamp(t_bag_start)}'
    ax.set_title(title)

    ax.set_xlabel('Time (s)')
    ax.legend()
    # y values times 1000 to convert to mm
    scale = 1000    
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale))
    ax.yaxis.set_major_formatter(ticks)
    ax.set_ylabel('Position (mm)')
    plt.tight_layout()
    keep_max_n_files(os.path.join(save_dir, 'position.png'))
    plt.savefig(os.path.join(save_dir, 'position.png'), dpi=300)
    if show_position:
        plt.show()
    else:
        plt.close()

    if show_3d and use_imu_isam2:
        # must add init pos back
        pos_data = pos_imu_isam 
        pos_data = pos_data + init_pos
        q_data = q_imu_isam
        # plot 3d position
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t_to_tip = [-0.00445981, -0.01745655, 0.14366956]
        
        pos_tip = []
        for pos, q in zip(pos_data, np.array(q_data).reshape(-1, 4)):
            R = transformations.quaternion_matrix(q)[:3, :3]
            pos_tip.append(pos + R @ t_to_tip)
        pos_tip = np.vstack(pos_tip)

        pos_to_plot = pos_tip

        ax.scatter(pos_to_plot[:, 0], pos_to_plot[:, 1], pos_to_plot[:, 2], s=3, c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Position')
        delta_x = np.max(pos_to_plot[:, 0]) - np.min(pos_to_plot[:, 0])
        delta_y = np.max(pos_to_plot[:, 1]) - np.min(pos_to_plot[:, 1])
        delta_z = np.max(pos_to_plot[:, 2]) - np.min(pos_to_plot[:, 2])
        # add biggest max delta + 10% margin
        max_range = np.max([delta_x, delta_y, delta_z]) * 1.1
        mid_x = (np.max(pos_to_plot[:, 0]) + np.min(pos_to_plot[:, 0])) * 0.5
        mid_y = (np.max(pos_to_plot[:, 1]) + np.min(pos_to_plot[:, 1])) * 0.5
        mid_z = (np.max(pos_to_plot[:, 2]) + np.min(pos_to_plot[:, 2])) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'position_3d.png'), dpi=300)
        plt.show()

    # plot for gyro bias
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    t = data_np[fusion_name][:, 0]
    ax.scatter(t, bw[:, 0], s=1)
    ax.scatter(t, bw[:, 1], s=1)
    ax.scatter(t, bw[:, 2], s=1)

    # if use_isam2:
    # plot gyro bias of ISAM as black dots
    t_isam = np.vstack(t_isam)
    bw_isam = np.vstack(bw_isam)
    ax.scatter(t_isam, bw_isam[:, 0], s=1, c='k')
    ax.scatter(t_isam, bw_isam[:, 1], s=1, c='k')
    ax.scatter(t_isam, bw_isam[:, 2], s=1, c='k')

    ax.set_title('Gyro bias')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bias (rad/s)')
    ax.legend(['x', 'y', 'z'])
    ax.grid()
    plt.tight_layout()
    keep_max_n_files(os.path.join(save_dir, 'gyro_bias.png'))
    plt.savefig(os.path.join(save_dir, 'gyro_bias.png'), dpi=300)
    plt.close()


    # do same for orientation (euler angles)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    draw_occlusions(ax, occlusions)
    ax.grid()
    if use_ots:
        euler_ots = np.vstack(euler_ots)
        for i, frame in enumerate(frames_ots_unique):
            c_ots = c_ots_list[i]
            if '_inactive' in frame:
                try:
                    i = frames_ots_unique.index(frame.replace('_inactive', ''))
                except ValueError:
                    pass
                c_ots = 'gray'
            m_ots = m_ots_list[i]
            idx = np.where(frames_ots == frame)[0]
            ax.scatter(t_ots[idx], euler_ots[idx, 0], s=s_ots, c=c_ots, marker=m_ots) if show_axes[0] else None
            ax.scatter(t_ots[idx], euler_ots[idx, 1], s=s_ots, c=c_ots, marker=m_ots) if show_axes[1] else None
            ax.scatter(t_ots[idx], euler_ots[idx, 2], s=s_ots, c=c_ots, marker=m_ots) if show_axes[2] else None
        
    if use_imu_rt:
        euler = euler_from_quaternion(q)
        ax.scatter(t, euler[:, 0], s=s_ori, label='roll', c=colors[1]) if show_axes[0] else None
        ax.scatter(t, euler[:, 1], s=s_ori, label='pitch', c=colors[3]) if show_axes[1] else None
        ax.scatter(t, euler[:, 2], s=s_ori, label='yaw', c=colors[5]) if show_axes[2] else None

    if use_isam2:
        euler_isam = euler_from_quaternion(q_isam)
        ax.scatter(t_isam, euler_isam[:, 0], s=4, c='r') if show_axes[0] else None
        ax.scatter(t_isam, euler_isam[:, 1], s=4, c='r') if show_axes[1] else None
        ax.scatter(t_isam, euler_isam[:, 2], s=4, c='r') if show_axes[2] else None
        # lw = 1
        # ax.plot(t_isam.reshape(-1), euler_isam[:, 0], c='r', label='roll (ISAM2)', linestyle='--', linewidth=lw)
        # ax.plot(t_isam.reshape(-1), euler_isam[:, 1], c='r', label='pitch (ISAM2)', linestyle='--', linewidth=lw)
        # ax.plot(t_isam.reshape(-1), euler_isam[:, 2], c='r', label='yaw (ISAM2)', linestyle='--', linewidth=lw)

    if use_imu_isam2:
        euler_imu_isam = euler_from_quaternion(q_imu_isam)
        ax.scatter(t_imu_isam, euler_imu_isam[:, 0], s=s_ori, c=colors[4], label='roll (IMU_ISAM2)') if show_axes[0] else None
        ax.scatter(t_imu_isam, euler_imu_isam[:, 1], s=s_ori, c=colors[0], label='pitch (IMU_ISAM2)') if show_axes[1] else None
        ax.scatter(t_imu_isam, euler_imu_isam[:, 2], s=s_ori, c=colors[2], label='yaw (IMU_ISAM2)') if show_axes[2] else None

    if use_hh_mm_ss:
        # tick_positions = ax.get_xticks()
        # create ticks every 10 seconds
        tick_positions = np.arange(t_bounds[0], t_bounds[1], 10)
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.set_xticklabels([seconds_to_timestamp(t) for t in tick_positions])
        # 45 degree rotation
        plt.xticks(rotation=45)

    ax.set_title(f'Orientation, start at {seconds_to_timestamp(t_bounds[0])}')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.set_ylabel('Angle (deg)')
    plt.tight_layout()
    keep_max_n_files(os.path.join(save_dir, 'orientation.png'))
    plt.savefig(os.path.join(save_dir, 'orientation.png'), dpi=300)
    if show_orientation:
        plt.show()
    print('Done plotting..')

