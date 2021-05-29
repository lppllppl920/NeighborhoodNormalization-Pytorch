from pathlib import Path
from skimage import measure
import numpy as np
import sys
import argparse

# if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import tqdm
import open3d as o3d
import gc

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

except Exception as err:
    print('Warning: %s' % (str(err)))
    print('Failed to import PyCUDA.')
    exit()


class TSDFVolume(object):
    def __init__(self, vol_bnds, voxel_size, trunc_margin):
        # Define voxel volume parameters
        self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)  # ensure C-order contiguous
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)  # ensure C-order contiguous
        # print("Voxel volume size: %d x %d x %d" % (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))

        # Initialize pointers to voxel volume in CPU memory
        # Assign oversized tsdf volume
        self._tsdf_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)  # -2.0 *
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(
            np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # Copy voxel volumes to GPU
        self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
        self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
        self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
        cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

        # Cuda kernel function (C++)
        self._cuda_src_mod_with_confidence_map = SourceModule("""
          __global__ void integrate(float * tsdf_vol,
                                    float * weight_vol,
                                    float * color_vol,
                                    float * vol_dim,
                                    float * vol_origin,
                                    float * cam_intr,
                                    float * cam_pose,
                                    float * other_params,
                                    float * color_im,
                                    float * depth_im) {

            // Get voxel index
            int gpu_loop_idx = (int) other_params[0];
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
            int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

            int vol_dim_x = (int) vol_dim[0];
            int vol_dim_y = (int) vol_dim[1];
            int vol_dim_z = (int) vol_dim[2];

            if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                return;

            // Get voxel grid coordinates (note: be careful when casting)
            float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
            float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
            float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

            // Voxel grid coordinates to world coordinates
            float voxel_size = other_params[1];
            float pt_x = vol_origin[0]+voxel_x*voxel_size;
            float pt_y = vol_origin[1]+voxel_y*voxel_size;
            float pt_z = vol_origin[2]+voxel_z*voxel_size;

            // World coordinates to camera coordinates
            float tmp_pt_x = pt_x-cam_pose[0*4+3];
            float tmp_pt_y = pt_y-cam_pose[1*4+3];
            float tmp_pt_z = pt_z-cam_pose[2*4+3];
            float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
            float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
            float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

            // Because of the long tube of endoscope, the minimum depth to consider is not zero
            float min_depth = other_params[6];
            if (cam_pt_z < min_depth) {
                return;
            }

            // Camera coordinates to image pixels
            int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
            int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);

            // Skip if outside view frustum
            int im_h = (int) other_params[2];
            int im_w = (int) other_params[3];
            if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
                return;

            // Skip invalid depth
            float depth_value = depth_im[pixel_y*im_w+pixel_x];

            float max_depth = other_params[8];
            if (depth_value <= 0.0 || depth_value > max_depth)
                return;

            float trunc_margin = other_params[4];
            //float depth_diff = depth_value - cam_pt_z;

            float diff = (depth_value - cam_pt_z) * sqrtf(1 + powf((cam_pt_x / cam_pt_z), 2) + powf((cam_pt_y / cam_pt_z), 2));
            if (diff <= -trunc_margin)
              return;

            //if (depth_diff < -trunc_margin)
            //    return;

            float slope = other_params[7];
            // float dist = fmin(1.0f, depth_diff / slope);
            float dist = fmin(1.0f, diff / slope);

            float w_old = weight_vol[voxel_idx];
            float obs_weight = other_params[5];
            float w_new = w_old + obs_weight;
            tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx] * w_old + dist * obs_weight) / w_new;
            weight_vol[voxel_idx] = w_new;


            // Integrate color
            float new_color = color_im[pixel_y * im_w + pixel_x];
            float new_b = floorf(new_color / (256 * 256));
            float new_g = floorf((new_color - new_b * 256 * 256) / 256);
            float new_r = new_color - new_b * 256 * 256 - new_g * 256;

            float old_color = color_vol[voxel_idx];
            float old_b = floorf(old_color / (256 * 256));
            float old_g = floorf((old_color - old_b * 256 * 256) / 256);
            float old_r = old_color - old_b * 256 * 256 - old_g * 256;

            new_b = fmin(roundf((old_b * w_old + new_b * obs_weight) / w_new), 255.0f);
            new_g = fmin(roundf((old_g * w_old + new_g * obs_weight) / w_new), 255.0f);
            new_r = fmin(roundf((old_r * w_old + new_r * obs_weight) / w_new), 255.0f);

            color_vol[voxel_idx] = new_b * 256 * 256 + new_g * 256 + new_r;
          }""")

        self._cuda_integrate = self._cuda_src_mod_with_confidence_map.get_function("integrate")
        # Determine block/grid size on GPU
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        # _n_gpu_loops specifies how many loops for the GPU to process the entire volume
        self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
            np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, min_depth, depth_slope, max_depth, obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[:, :, 2] * 256 * 256 + color_im[:, :, 1] * 256 + color_im[:, :, 0])

        # integrate voxel volume (calls CUDA kernel)
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(self._tsdf_vol_gpu,
                                 self._weight_vol_gpu,
                                 self._color_vol_gpu,
                                 cuda.InOut(self._vol_dim.astype(np.float32)),
                                 cuda.InOut(self._vol_origin.astype(np.float32)),
                                 cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                 cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                 cuda.InOut(np.asarray(
                                     [gpu_loop_idx, self._voxel_size, im_h, im_w, self._trunc_margin,
                                      obs_weight, min_depth, depth_slope, max_depth],
                                     np.float32)),
                                 cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                 cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                 block=(self._max_gpu_threads_per_block, 1, 1), grid=(
                    int(self._max_gpu_grid_dim[0]), int(self._max_gpu_grid_dim[1]),
                    int(self._max_gpu_grid_dim[2])))

    # Copy voxel volume to CPU
    def get_volume(self):
        cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    # Get mesh of voxel volume via marching cubes
    def get_mesh(self, only_visited=False):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0, gradient_direction='ascent')

        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates

        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
        colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
        colors = np.transpose(np.uint8(np.floor(np.asarray([colors_r, colors_g, colors_b])))).reshape(-1, 3)

        if only_visited:
            verts_indxes = verts_ind[:, 0] * weight_vol.shape[1] * weight_vol.shape[2] + \
                           verts_ind[:, 1] * weight_vol.shape[2] + verts_ind[:, 2]
            weight_vol = weight_vol.reshape((-1))
            valid_vert_indexes = np.nonzero(weight_vol[verts_indxes] >= 1)[0]
            valid_vert_indexes = set(valid_vert_indexes)

            indicators = np.array([face in valid_vert_indexes for face in faces[:, 0]]) \
                         & np.array([face in valid_vert_indexes for face in faces[:, 1]]) \
                         & np.array([face in valid_vert_indexes for face in faces[:, 2]])

            return verts, faces[indicators], norms, colors

        return verts, faces, norms, colors


def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array(
        [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0, 0],
                               (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array(
                                   [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1, 1],
                               np.array([0, max_depth, max_depth, max_depth, max_depth])])
    view_frust_pts = np.dot(cam_pose[:3, :3], view_frust_pts) + np.tile(cam_pose[:3, 3].reshape(3, 1), (
        1, view_frust_pts.shape[1]))  # from camera to world coordinates
    return view_frust_pts


def read_intrinsics(file_path):
    intrinsics = np.zeros((3, 3), dtype=np.float32)
    with open(str(file_path), "r") as fp:
        for i in range(3):
            line = fp.readline()
            words = line.split(sep="\t")
            intrinsics[i][0] = float(words[0])
            intrinsics[i][1] = float(words[1])
            intrinsics[i][2] = float(words[2])

    intrinsics = intrinsics / intrinsics[2, 2]
    return intrinsics


def read_pose(file_path):
    pose = np.zeros((4, 4), dtype=np.float32)
    with open(str(file_path), "r") as fp:
        for i in range(4):
            line = fp.readline()
            words = line.split(sep="\t")
            pose[i][0] = float(words[0])
            pose[i][1] = float(words[1])
            pose[i][2] = float(words[2])
            pose[i][3] = float(words[3])

    # Some poses are not normalized
    pose = pose / pose[3, 3]
    return pose


def determine_volume_size(depth_path_list, max_depth, camera_intrinsics, camera_pose_list):
    vol_bnds = np.zeros((3, 2))
    n_imgs = len(depth_path_list)
    depth_map = cv2.imread(str(depth_path_list[0]), cv2.IMREAD_ANYDEPTH)
    height, width = depth_map.shape[:2]
    for i in range(n_imgs):
        depth_map = cv2.imread(str(depth_path_list[i]), cv2.IMREAD_ANYDEPTH) / 1000.0
        depth_map = depth_map.reshape((height, width, 1))
        # To avoid having too large region to fuse
        depth_map = np.clip(depth_map, a_min=0.0, a_max=max_depth)
        if np.any(np.isnan(camera_pose_list[i])):
            continue
        view_frust_pts = get_view_frustum(depth_map, camera_intrinsics, camera_pose_list[i])
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    return vol_bnds


'''
Parameters from 3DMatch fusion code
  int base_frame_idx = 8;
  int first_frame_idx = 8;
  float num_frames = 50;

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;
'''


def tsdf_fusion(sequence_path, output_path):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    camera_intrinsics_path = sequence_path.parent / "camera-intrinsics.txt"
    camera_intrinsics = read_intrinsics(file_path=camera_intrinsics_path)

    frame_count = 50
    max_depth = 6.0
    voxel_size = 0.01
    factor = 5.0
    trunc_margin = voxel_size * factor
    max_voxel_count = 256e6
    print(f"Processing {str(sequence_path)}")
    # Read depth, color, and camera pose. Use tsdf to fuse all depth
    total_depth_path_list = sorted(list(sequence_path.glob("*.depth.png")))  # 16-bit int in millimeter
    total_pose_path_list = sorted(list(sequence_path.glob("*.pose.txt")))  # camera-to-world in meter
    total_color_path_list = sorted(list(sequence_path.glob("*.color.png")))
    assert (len(total_depth_path_list) == len(total_pose_path_list))

    if len(total_color_path_list) == 0:
        has_color = False
    else:
        has_color = True

    n_imgs = len(total_depth_path_list)

    total_camera_pose_list = list()
    for i in range(n_imgs):
        pose = read_pose(str(total_pose_path_list[i]))
        total_camera_pose_list.append(pose)

    result_list = sorted(list(output_path.glob("cloud_bin*.ply")))
    if len(result_list) >= int(np.floor(n_imgs / frame_count)):
        print(f"Sequence {str(sequence_path)} already processed")
        return

    color_path_list = None
    tq = tqdm.tqdm(total=int(np.floor(n_imgs / frame_count)))
    for j in range(int(np.floor(n_imgs / frame_count))):
        fused_model_path = output_path / f"cloud_bin_{j}.ply"
        if fused_model_path.exists():
            tq.update(1)
            continue
        curr_voxel_size = voxel_size
        depth_path_list = total_depth_path_list[j * frame_count: (j + 1) * frame_count]
        camera_pose_list = total_camera_pose_list[j * frame_count: (j + 1) * frame_count]
        if has_color:
            color_path_list = total_color_path_list[j * frame_count: (j + 1) * frame_count]
        color_img = None
        volume_bounds = determine_volume_size(depth_path_list=depth_path_list, camera_intrinsics=camera_intrinsics,
                                              camera_pose_list=camera_pose_list, max_depth=max_depth)

        vol_dim = np.ceil((volume_bounds[:, 1] - volume_bounds[:, 0]) / curr_voxel_size)
        if np.prod(vol_dim) > max_voxel_count:
            print(f"Changing voxel size from {curr_voxel_size} to "
                  f"{curr_voxel_size / (max_voxel_count / np.prod(vol_dim)) ** (1.0 / 3)}")
            curr_voxel_size = curr_voxel_size / ((max_voxel_count / np.prod(vol_dim)) ** (1.0 / 3))
            trunc_margin = curr_voxel_size * factor

        tsdf_vol = TSDFVolume(vol_bnds=volume_bounds, voxel_size=curr_voxel_size,
                              trunc_margin=trunc_margin)
        for i in range(frame_count):
            depth_map = cv2.imread(str(depth_path_list[i]), cv2.IMREAD_ANYDEPTH) / 1000.0
            if has_color:
                color_img = cv2.imread(str(color_path_list[i]), cv2.IMREAD_COLOR)
            elif color_img is None:
                color_img = np.ones((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)
            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_img, depth_map, camera_intrinsics, camera_pose_list[i],
                               min_depth=0.0,
                               obs_weight=1.,
                               depth_slope=trunc_margin,
                               max_depth=max_depth)
        verts, faces, norms, colors = tsdf_vol.get_mesh(only_visited=True)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts).reshape((-1, 3)))
        mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(-norms).reshape((-1, 3)))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces).reshape((-1, 3)))

        colors = np.asarray(colors / 255.0).reshape((-1, 3))
        colors = np.concatenate(
            [colors[:, 2].reshape((-1, 1)), colors[:, 1].reshape((-1, 1)), colors[:, 0].reshape((-1, 1))], axis=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Do some clean-up
        mesh = mesh.remove_non_manifold_edges()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(str(fused_model_path), mesh)

        del tsdf_vol
        gc.collect()
        tq.update(1)

    tq.close()


def main():
    parser = argparse.ArgumentParser(
        description='3DMatch Mesh Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, required=True, help='path to the 3DMatch image data')
    parser.add_argument('--output_root', type=str, required=True, help='path to the 3DMatch mesh output')
    args = parser.parse_args()

    sequence_path_list = sorted(list(Path(args.data_root).rglob("seq-*")))

    for sequence_path in sequence_path_list:
        tsdf_fusion(sequence_path, Path(args.output_root) / sequence_path.parents[0].name / sequence_path.name)


if __name__ == "__main__":
    main()
