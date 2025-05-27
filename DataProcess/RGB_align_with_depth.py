import pickle
import os
import sys
sys.path.append( '/root/autodl-tmp/AttieCode')
from Config.config import CONF
import numpy as np
from PIL import Image
import cv2
from utils.read_intrinsic import read_intrinsic
from cv2 import rgbd 

os.environ["QT_QPA_PLATFORM"] = "xcb"

# print(sys.path)
# from utils.read_pgm_p5 import pgmread_p5

# Map_matrix = K_c @ np.linalg.inv(K_d)
                
def align_rgb_with_depth(scan_path): #This is used for all frames in a scan
    scan_id = scan_path.split('/')[-1]
    intrinsic_path = os.path.join(scan_path, 'sequence', '_info.txt')
    RGB_intrinsic_info = read_intrinsic(intrinsic_path, mode='rgb')
    Depth_intrinsic_info = read_intrinsic(intrinsic_path, mode='pgm')
    
    frame_size = RGB_intrinsic_info['m_frames_size']
    # 1) read intrinsics from  _info.txt
    Depth_intrinsic = Depth_intrinsic_info['m_intrinsic'][:3,:3].reshape(3,3).astype(np.float32)
    RGB_intrinsic = RGB_intrinsic_info['m_intrinsic'][:3,:3].reshape(3,3).astype(np.float32)
    # Depth_intrinsic = np.array([[176.594, 0, 114.613],
    #                 [0, 240.808, 85.7915],
    #                 [0, 0, 1]]).astype(np.float32)
    # RGB_instrinsic = np.array([[756.832, 0, 492.889],
    #                 [0, 756.026, 270.419],
    #                 [0, 0, 1]]).astype(np.float32)
    # Depth→RGB extrinsic (3×4 float32)
    Rt = np.hstack((np.eye(3, dtype=np.float32),
                    np.zeros((3,1), dtype=np.float32)))  # :contentReference[oaicite:6]{index=6}

    for i in range(frame_size):
        jpg_name = 'frame-'+ str(i).zfill(6)+'.color.jpg'
        pgm_name = 'frame-' + str(i).zfill(6)  + '.depth.pgm' 

        RGB_path = os.path.join(scan_path, 'sequence', jpg_name)
        Depth_path = os.path.join(scan_path, 'sequence', pgm_name)
        
        H_d, W_d = Depth_intrinsic_info['m_Height'], Depth_intrinsic_info['m_Width']
        Depth_raw = cv2.imread(Depth_path, cv2.IMREAD_UNCHANGED)
        # print(Depth_path, RGB_path)
        assert Depth_raw is not None and Depth_raw.ndim == 2, "Depth must be a single-channel image"
        if Depth_raw.dtype == np.uint16:
            Depth_raw = Depth_raw.astype(np.float32) / 1000.0
        else:
            Depth_raw = Depth_raw.astype(np.float32)
        
        # RGB_dim = np.array([RGB_height, RGB_width,RGB_Channel])
        H_c, W_c = RGB_intrinsic_info['m_Height'], RGB_intrinsic_info['m_Width']
        RGB_raw = cv2.imread(RGB_path, cv2.IMREAD_COLOR)#.reshape(-1).reshape(RGB_dim)
        assert RGB_raw is not None and RGB_raw .ndim == 3, "Failed to load color image"
        
        Rt = np.hstack((np.eye(3, dtype=np.float32), np.zeros((3,1), dtype=np.float32))) 

        # 1.4 Back-project depth to 3D, transform & project into color
        pts3d = cv2.rgbd.depthTo3d(Depth_raw, Depth_intrinsic)        # (H_d×W_d×3) :contentReference[oaicite:0]{index=0}
        pts   = pts3d.reshape(-1,3).T                # (3,N)
        ptsc  = Rt[:, :3] @ pts + Rt[:, 3:4]         # (3,N)
        uvz   = RGB_intrinsic @ ptsc                           # (3,N)
        uv    = uvz[:2] / uvz[2:3]                   # normalize :contentReference[oaicite:1]{index=1}

        # 1.5 Build remap maps and warp
        map_x = uv[0].reshape(H_d, W_d).astype(np.float32)
        map_y = uv[1].reshape(H_d, W_d).astype(np.float32)
        aligned_rgb = cv2.remap(
            RGB_raw, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )  # shape = H_d×W_d×3, holes = (0,0,0) :contentReference[oaicite:2]{index=2}

        # 1.6 Postprocess aligned RGB map (fill holes)
        aligned_bgr = aligned_rgb  # this is BGR already
        
        # mask = 1 for holes, 0 everywhere else
        hole_mask = np.all(aligned_bgr == 0, axis=2).astype(np.uint8)  # shape = H_d×W_d
        # cv2.inpaint requires 8-bit 1-channel mask, and 8-bit/color source
        # Convert aligned_rgb → 8-bit if needed
        aligned_8u = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        if aligned_8u.dtype != np.uint8:
            print("not uint8")
            aligned_8u = np.clip(aligned_8u, 0, 255).astype(np.uint8)

        filled_bgr = cv2.inpaint(
            aligned_8u, 
            hole_mask, 
            inpaintRadius=3, 
            flags=cv2.INPAINT_TELEA
        )  # :contentReference[oaicite:3]{index=3}

        filled_rgb = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)

        # cv2.imshow('Aligned Color Image', aligned_rgb)
        # cv2.imshow('raw Image', Depth_raw)
        # cv2.imshow('filled Image', filled_rgb)
        # cv2.waitKey(10000)  
        # cv2.destroyAllWindows()

        # Save the image
        image_folder = os.path.join( CONF.PATH.R3SCAN_DATA_OUT, 'aligned_RGB' , scan_id)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        aligned_rgb_name = 'frame-' + str(i).zfill(6) + '.aligned_rgb.jpg'
        output_path = os.path.join( image_folder, aligned_rgb_name)
        print(output_path)
        success = cv2.imwrite(output_path, filled_rgb)
        if success:
            print("Saved")
        # print(f"Image saved to {output_path}")

if __name__ == "__main__":
    for scan in os.listdir(CONF.PATH.R3SCAN_RAW):
        if os.path.isdir( os.path.join(CONF.PATH.R3SCAN_RAW, scan) ) and scan != '3DSSG_subset':
            sequence_folder = os.path.join( CONF.PATH.R3SCAN_RAW, scan)
            if "0cac75b1-8d6f-2d13-8c17-9099db8915bc" in sequence_folder:
                align_rgb_with_depth(sequence_folder)
         


