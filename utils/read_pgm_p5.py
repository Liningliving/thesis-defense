import numpy as np
import sys
# from numpy import array, uint16, fromfile, array
np.set_printoptions(threshold=sys.maxsize)

filename = "/mnt/hgfs/vm_shared_files/f62fd5ff-9a3f-2f44-894c-462b3997d695/sequence/frame-000001.depth.pgm"

def pgmread_p5(filename, dtype=np.uint16):
    """Reads a PGM image file and returns a numpy array."""
    with open(filename, 'rb') as f:
        width, height = 0, 0
        magicNum = ''
        maxVal = 0
        count = 0

        # Read header lines, ignoring comments
        while count < 3:
            line = f.readline().decode('ascii', errors='ignore').strip()
            print(line)
            if line.startswith('#'):
                continue
            count += 1
            if count == 1:
                magicNum = line
                if magicNum != 'P5':
                    raise ValueError("Unsupported format (must be P5)")
            elif count == 2:
                width, height = map(int, line.split())
            elif count == 3:
                maxVal = int(line)

        # Skip any remaining whitespace after the header
        # f.read(1)  # Skip the newline after maxVal

        # Read pixel data
        pixel_data = np.fromfile(f, dtype, count=width * height)
        pixel_data = pixel_data.reshape((height, width))

    return np.array(pixel_data)

if __name__ == "__main__":  
    depth_np_array = pgmread_p5(filename)
    print(depth_np_array)
    # print("Image read successfully!")
    # print("Image shape:", depth_np_array.shape)
    print("Image data type:", depth_np_array.dtype)