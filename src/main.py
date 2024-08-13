from src.SIV_library.processing import Video, Processor, Viewer
from src.SIV_library.lib import SIV, OpticalFlow

import numpy as np
import torch
import matplotlib.pyplot as plt

import os


if __name__ == "__main__":
    video_file = "IMG_3010_full.MOV"
    # video_file = r"plume simulation.MOV"
    fn = video_file.split(".")[0]

    # reference frame specified first, then the range we want to analyse with SIV
    # frames = [0, *(i for i in range(int(11*30), int(33*30)))] # Change range(start, end)
    frames = [*(i for i in range(int(11*30), int(33*30)))] # Change range(start, end)

    vid = Video(rf"Test Data/{video_file}", df='.png', indices=frames)
    vid.create_frames()

    processor = Processor(rf"Test Data/{fn}", df='.png')
    processor.postprocess()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    capture_fps = 30. # 240.
    scale = 1 # 
    
    calibration_m_to_pixels = 2./2000.
    calibration_sec_to_frame = 1/capture_fps 

    siv = SIV(
        folder=rf"Test Data/{fn}_PROCESSED",
        device=device,
        mode=1,
        window_size=64,
        overlap=0,
        search_area=(32, 32, 32, 32),
        num_passes = 2,
    )
    
    opt_flow = OpticalFlow(
        folder = rf"Test Data/{fn}_PROCESSED",
        device = device,
        alpha = 10,
        grid_size_avg = 16,
        )


    # if f"{fn}_WS_{siv.window_size}_O_{siv.overlap}_SA_{siv.search_area[0]}_NP_{siv.num_passes}_RESULTS" not in os.listdir(f"Test Data"):
        # x, y, vx, vy = siv.run(mode=1) # Commented by Sagar
        ## Sagar start
        # res = []
    
    results_save_dir = rf"Test Data/{fn}_WS_{siv.window_size}_O_{siv.overlap}_SA_{siv.search_area[0]}_NP_{siv.num_passes}_RESULTS"
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    else:
        print(rf"{results_save_dir} exists")
    
    count = 0
    for x, y, vx, vy in siv():
        # Append the results from each iteration to the respective lists
        # res.append((x.cpu().numpy(),y.cpu().numpy(),vx.cpu().numpy(),vy.cpu().numpy()))
        res = np.stack((x.cpu().numpy(),y.cpu().numpy(),vx.cpu().numpy(),vy.cpu().numpy()),axis=0)
        np.savez(rf"{results_save_dir}/{count}", res = res, calibration_m_to_pixels = calibration_m_to_pixels, calibration_sec_to_frame = calibration_sec_to_frame)
        count+=1
                
    # res = np.array(res)
    ## Sagar end
        
        # res = np.array((x.cpu(), y.cpu(), vx.cpu(), vy.cpu())) # Commented by Sagar
    # np.save(rf"Test Data/{fn}_WS_{siv.window_size}_O_{siv.overlap}_SA_{siv.search_area[0]}_NP_{siv.num_passes}_RESULTS", res) # Commented by Sagar
    # np.savez(rf"Test Data/{fn}_WS_{siv.window_size}_O_{siv.overlap}_SA_{siv.search_area[0]}_NP_{siv.num_passes}_RESULTS", res = res, calibration_m_to_pixels = calibration_m_to_pixels, calibration_sec_to_frame = calibration_sec_to_frame)

    # else:
    print("Loading results...")
        # res = np.load(rf"Test Data/{fn}_WS_{siv.window_size}_O_{siv.overlap}_SA_{siv.search_area[0]}_NP_{siv.num_passes}_RESULTS.npy") # Commented by Sagar.
    len_res_files = len(os.listdir(rf"{results_save_dir}"))
    shape_res_file = res.shape
    res_all = np.empty((len_res_files,*shape_res_file))
    for i in range(len_res_files):    
        data = np.load(rf"{results_save_dir}/{i}.npz")
        res_data = data['res']
        calibration_m_to_pixels = data['calibration_m_to_pixels']
        calibration_sec_to_frame = data['calibration_sec_to_frame']
        res_all[i,:,:,:] = res_data

    viewer = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=1., capture_fps=capture_fps, siv_window_size = siv.window_size, siv_overlap = siv.overlap, siv_search_area = siv.search_area, siv_num_passes = siv.num_passes, optical_flow_flag = False)

    # viewer.play_video()
    viewer.vector_field(res_all, calibration_m_to_pixels, calibration_sec_to_frame)
    # viewer.velocity_field(res, scale, 30, 'cubic')
    
    # The below algorithm needs to be corrected for the new individual result file storage way.
    # Running algorithm for optical flow.
    if f"{fn}_Optical_flow_RESULTS.npy" not in os.listdir(f"Test Data"):
        res_opt = []
        for x, y, vx, vy in opt_flow():
            res_opt.append((x.cpu().numpy(),y.cpu().numpy(),vx.cpu().numpy(),vy.cpu().numpy()))
        res_opt = np.array(res_opt)

        np.save(rf"Test Data/{fn}_Optical_flow_RESULTS", res_opt)
        
    else:
        print("Loading Optical flow results...")
        res_opt = np.load(rf"Test Data/{fn}_Optical_flow_RESULTS.npy")
        
    viewer_opt_flow = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=1., capture_fps=capture_fps, optical_flow_flag = True)
    viewer_opt_flow.vector_field(res_opt, scale)   
    
    
    # Code for calculating mean velocity.
    
    x = res_all[0, 0, :, :]*calibration_m_to_pixels
    y = res_all[0, 1, :, :]*calibration_m_to_pixels
    vx_mean = np.mean(res_all[:, 2, :, :], axis=0)*calibration_m_to_pixels/calibration_sec_to_frame
    vy_mean = np.mean(res_all[:, 3, :, :], axis=0)*calibration_m_to_pixels/calibration_sec_to_frame
    v_magnitude = np.sqrt(vx_mean**2 + vy_mean**2)
    
    fig, ax = plt.subplots()
    vectors = ax.quiver(x, np.flip(y, axis = 0), vx_mean, vy_mean, v_magnitude, cmap='jet') # Flipping y axis vertically so that y = 0 is at the bottom.
    # ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(vectors, ax = ax)
    cbar.set_label('Magnitude')
    
    ax.set_title('Mean velocity (m/s)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    

# Measure the velocity with a probe and check the difference.
# Capture a background image with very dark background.
