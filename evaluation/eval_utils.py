import numpy as np
import torch
import MinkowskiEngine as ME
import pyacvd, pyvista
import open3d as o3d
import os


def summarize_resolution_mismatch_results(root, inlier_ratio_threshold_list):
    mean_results_over_scenes = dict()

    pair_result_path_list = sorted(list(root.rglob("voxel_size_pair_*")))
    for pair_result_path in pair_result_path_list:
        pair_result_path_str = str(pair_result_path)
        ind = pair_result_path_str.find("voxel_size_pair_")
        ind2 = pair_result_path_str.find("_", ind + len("voxel_size_pair_"))

        voxel_size_0 = float(pair_result_path_str[ind + len("voxel_size_pair_"):ind2])
        voxel_size_1 = float(pair_result_path_str[ind2 + 1:])

        npy_path_list = sorted(list(pair_result_path.glob("*.npy")))
        results_list_all_scene = list()
        for npy_path in npy_path_list:
            result_array_per_scene = np.load(str(npy_path))
            result_list_per_scene_all_taus = list()
            for inlier_ratio_threshold in inlier_ratio_threshold_list:
                result_list_per_scene_all_taus.append(
                    np.mean((result_array_per_scene[:, 0] > inlier_ratio_threshold)))
            results_per_scene_all_taus = np.asarray(result_list_per_scene_all_taus)
            results_list_all_scene.append(results_per_scene_all_taus)
        mean_results_over_scenes[(voxel_size_0, voxel_size_1)] = \
            np.mean(np.asarray(results_list_all_scene), axis=0)

    return mean_results_over_scenes


def extract_mesh_features(o3d_mesh, tgt_edge_length, num_cluster, subdivide_factor, default_edge_length):
    mesh = from_o3d_to_pyvista_mesh(o3d_mesh)
    mesh_cluster = pyacvd.Clustering(mesh)

    if mesh.n_faces < subdivide_factor * num_cluster:
        ratio = int(np.ceil(subdivide_factor * num_cluster / (4.0 * mesh.n_faces)))
        mesh_cluster.subdivide(nsub=ratio)

    mesh_cluster.cluster(num_cluster)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)

    temp_length = calc_edge_length_vista(remeshed_model)
    correct_ratio = (tgt_edge_length / temp_length) ** 2

    mesh_cluster.cluster(num_cluster / correct_ratio)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)
    o3d_mesh = from_pyvista_to_o3d_mesh(remeshed_model)

    # Do the cleaning and repairing here!
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()
    o3d_mesh = o3d_mesh.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh = o3d_mesh.compute_vertex_normals(normalized=True)
    avg_edge_length = calc_edge_length(o3d_mesh)

    # Rescale the o3d mesh to default edge length by scaling its vertex values
    scale = default_edge_length / avg_edge_length
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices) * scale)
    avg_edge_length = calc_edge_length(o3d_mesh)

    return o3d_mesh, avg_edge_length, scale


def extract_features(model,
                     xyz,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False):
    '''
    xyz is a N x 3 matrix
    rgb is a N x 3 matrix and all color must range from [0, 1] or None
    normal is a N x 3 matrix and all normal range from [-1, 1] or None

    if both rgb and normal are None, we use Nx1 one vector as an input

    if device is None, it tries to use gpu by default

    if skip_check is True, skip rigorous checks to speed up

    model = model.to(device)
    xyz, feats = extract_features(model, xyz)
    '''

    if not skip_check:
        assert xyz.shape[1] == 3
        N = xyz.shape[0]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = []
    feats.append(np.ones((len(xyz), 1)))
    feats = np.hstack(feats)

    selected_indexes = ME.utils.sparse_quantize(xyz, return_index=True, quantization_size=voxel_size)[1]
    unique_xyz = xyz[selected_indexes]
    coords = np.floor(unique_xyz / voxel_size)
    coords = ME.utils.batched_coordinates([coords])
    feats = feats[selected_indexes]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = coords.int()

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    return unique_xyz, model(stensor).F.cpu().numpy()


def remesh_surface(o3d_mesh, tgt_edge_length, num_cluster, subdivide_factor):
    mesh = from_o3d_to_pyvista_mesh(o3d_mesh)
    mesh_cluster = pyacvd.Clustering(mesh)

    if mesh.n_faces < subdivide_factor * num_cluster:
        ratio = int(np.ceil(subdivide_factor * num_cluster / (4.0 * mesh.n_faces)))
        mesh_cluster.subdivide(nsub=ratio)

    mesh_cluster.cluster(num_cluster)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)

    temp_length = calc_edge_length_vista(remeshed_model)
    correct_ratio = (tgt_edge_length / temp_length) ** 2

    mesh_cluster.cluster(num_cluster / correct_ratio)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)
    o3d_mesh = from_pyvista_to_o3d_mesh(remeshed_model)

    # Do the cleaning and repairing here
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()
    o3d_mesh = o3d_mesh.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh = o3d_mesh.compute_vertex_normals(normalized=True)
    avg_edge_length = calc_edge_length(o3d_mesh)

    return o3d_mesh, avg_edge_length


def remesh_model(o3d_mesh, voxel_size, default_voxel_size):
    area = calc_total_surface_area(o3d_mesh)
    # number of cluster in pyacvd is the number of mesh triangles
    num_cluster = area / (np.sqrt(3) / 4 * voxel_size ** 2)
    mesh = from_o3d_to_pyvista_mesh(o3d_mesh)
    mesh_cluster = pyacvd.Clustering(mesh)

    if mesh.n_faces < 3.0 * num_cluster:
        ratio = int(np.ceil(3.0 * num_cluster / (4.0 * mesh.n_faces)))
        mesh_cluster.subdivide(nsub=ratio)

    mesh_cluster.cluster(num_cluster)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)

    temp_length = calc_edge_length_vista(remeshed_model)
    correct_ratio = (voxel_size / temp_length) ** 2

    mesh_cluster.cluster(num_cluster / correct_ratio)
    remeshed_model = mesh_cluster.create_mesh(flipnorm=False)
    o3d_mesh = from_pyvista_to_o3d_mesh(remeshed_model)

    # Do the cleaning and repairing here!
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()
    o3d_mesh = o3d_mesh.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh = o3d_mesh.compute_vertex_normals(normalized=True)
    avg_edge_length = calc_edge_length(o3d_mesh)
    # TODO: the o3d_mesh has already been scaled s0 that the edge length equals the default_voxel_size
    scale = default_voxel_size / avg_edge_length
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices) * scale)
    avg_edge_length = calc_edge_length(o3d_mesh)

    return o3d_mesh, avg_edge_length, scale


def from_o3d_to_pyvista_mesh(o3d_mesh):
    mesh = pyvista.PolyData()
    mesh.points = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    mesh.faces = np.concatenate([3 * np.ones((faces.shape[0], 1)), faces], axis=1).astype(np.int32)
    return mesh


def from_pyvista_to_o3d_mesh(vista_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vista_mesh.points))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(vista_mesh.faces).reshape((-1, 4))[..., 1:])
    return o3d_mesh


def calc_edge_length_vista(vista_mesh):
    faces = np.asarray(vista_mesh.faces).reshape((-1, 4))[:, 1:]
    vertices = np.asarray(vista_mesh.points)
    avg_edge_length = 1.0 / 3 * \
                      (np.mean(
                          np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))) +
                       np.mean(
                           np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))) +
                       np.mean(np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))))
    return avg_edge_length


def calc_edge_length(o3d_mesh):
    faces = np.asarray(o3d_mesh.triangles)
    vertices = np.asarray(o3d_mesh.vertices)
    avg_edge_length = 1.0 / 3 * \
                      (np.mean(
                          np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))) +
                       np.mean(
                           np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))) +
                       np.mean(np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))))
    return avg_edge_length


def calc_total_surface_area(o3d_mesh):
    faces = np.asarray(o3d_mesh.triangles)
    vertices = np.asarray(o3d_mesh.vertices)
    side_0 = np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]) ** 2, axis=1))
    side_1 = np.sqrt(np.sum((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]) ** 2, axis=1))
    side_2 = np.sqrt(np.sum((vertices[faces[:, 0], :] - vertices[faces[:, 2], :]) ** 2, axis=1))
    p = 0.5 * (side_0 + side_1 + side_2)
    surface_area = np.sum(np.sqrt(p * (p - side_0) * (p - side_1) * ((p - side_2))))
    return surface_area


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
               "pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename, dim=4):
    traj = []
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(dim, dim))
            for i in range(dim):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
        return traj
