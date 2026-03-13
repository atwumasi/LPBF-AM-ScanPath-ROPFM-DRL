
import time
start_time = time.time()  # Record the start time
import os
import sys
import numpy as np
import torch
import meshio
from model import UNET
from yaml_parser import pf_parse

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {DEVICE}")

# load model
case_name = "large"
MAIN_DIR = "/home/atwumasi/Documents/Austine/training_8November/ML_micro-main/3_predict"
ML_filename = "/home/atwumasi/Documents/Austine/training_8November/ML_micro-main/2_train/runs/LR0.0001_BS50_NE10_TS42000_VS5600/model_state_dict_9"
model = UNET(in_channels=3, out_channels=20).cuda()
model.load_state_dict(torch.load(os.path.join(MAIN_DIR, ML_filename)))
model.eval()

def read_path(pf_args):
    x_corners = pf_args['laser_path']['x_pos']
    y_corners = pf_args['laser_path']['y_pos']
    z_corners = pf_args['laser_path']['z_pos']
    power_control = pf_args['laser_path']['switch'][:-1]

    ts, xs, ys, zs, ps, mov_dir = [], [], [], [], [], []
    t_pre = 0.
    for i in range(len(x_corners) - 1):
        moving_direction = np.array([x_corners[i + 1] - x_corners[i], 
                                      y_corners[i + 1] - y_corners[i],
                                      z_corners[i + 1] - z_corners[i]])
        traveled_dist = np.linalg.norm(moving_direction)
        unit_direction = moving_direction/traveled_dist
        traveled_time = traveled_dist/pf_args['vel']
        ts_seg = np.arange(t_pre, t_pre + traveled_time, pf_args['dt'])
        xs_seg = np.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = np.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        zs_seg = np.linspace(z_corners[i], z_corners[i + 1], len(ts_seg))
        ps_seg = np.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        zs.append(zs_seg)
        ps.append(ps_seg)
        mov_dir.append(np.repeat(unit_direction[None, :], len(ts_seg), axis=0))
        t_pre = t_pre + traveled_time

    ts, xs, ys, zs, ps, mov_dir = np.hstack(ts), np.hstack(xs), np.hstack(ys), np.hstack(zs), np.hstack(ps), np.vstack(mov_dir)  
    return ts, xs, ys, zs, ps, mov_dir

def get_T_laser(pf_args, centroids, x_laser, y_laser, z_laser, power, unit_mov_dir):
    """Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
    """
    Q = 25 * power
    alpha = 5.2e-6
    kappa = 27
    X = centroids[:, 0] - x_laser
    Y = centroids[:, 1] - y_laser
    Z = centroids[:, 2] - z_laser
    R = np.sqrt(X**2 + Y**2 + Z**2)
    projection = X*unit_mov_dir[0] + Y*unit_mov_dir[1] + Z*unit_mov_dir[2]
    T = pf_args['T_ambient'] + Q / (2 * np.pi * kappa) / R * np.exp(-pf_args['vel'] / (2*alpha) * (R + projection))
    T = np.where(T > 2000., 2000., T)
    return T[:, None]

def rve_picker(x, y, z, large_array):
    rve = np.zeros((80, 80, 32))
    rve = large_array[x-40:x+40, y-40:y+40, z-32:z]
    return rve

def large_arr_updater(x, y, z, rve, large_array):
    large_array[x-40:x+40, y-40:y+40, z-32:z] = rve
    return large_array

def predictor(t_image):
    model.eval()
    x = t_image
    x[:, 0, :, :, :] = x[:, 0, :, :, :] 
    x[:, 1:, :, :, :] = x[:, 1:, :, :, :] 

    x = x.cuda()
    with torch.no_grad():
        output = model(x)
        preds = torch.argmax(output, dim=1)
        #predss = preds.to("cpu").detach().numpy()
        #predss = predss.squeeze(0)
    return preds.squeeze(0)#, predss
    
# initialize
skip = 100
max_t = 60000 #12402 #17203 #14821 #40950 #13650
Nx = 464
Ny = 464
Nz = 46
cell_size = 1e-3/Nx  #422.0  ### Always change this depending on the size on NX
div = 1/cell_size #422000 # pixels to meters needs modification all the time depending on the size on NX
x_laser = 0.085e-3 # 40*cell_size
y_laser = 0.085e-3 #40*cell_size
z_laser =0.065e-3 #32*cell_sizeb

pf_args = pf_parse(os.path.join(MAIN_DIR, '3_track.yaml'))
centroids = np.load(os.path.join(MAIN_DIR, '808032_centroids.npy')) #for local temperature field
large_array = np.load(os.path.join(MAIN_DIR, 'large_initial.npy'))
large_array = large_array.reshape((Nx, Ny, Nz), order="F")
t_large_array = torch.Tensor(large_array)
empty_array= torch.full((Nx, Ny, Nz), float('nan'))

t_image = torch.zeros(3, 80, 80, 32)
t_image = t_image[None, :, :, :, :]

ts, xs, ys, zs, ps, mov_dir = read_path(pf_args)
x = round(xs[0]/cell_size)
y = round(ys[0]/cell_size)
z = round((zs[0])/cell_size)
### (len(VOI)/2)*40
T0 = get_T_laser(pf_args, centroids, x_laser, y_laser, z_laser, ps[0], mov_dir[0]) 
T1 = get_T_laser(pf_args, centroids, x_laser, y_laser, z_laser, ps[0+skip], mov_dir[0+skip])  
T0 = T0.reshape((80, 80, 32), order="F")
T1 = T1.reshape((80, 80, 32), order="F")

I0 = rve_picker(x, y, z, large_array)
I0 = I0[None, None, :, :, :]
T0 = T0[None, None, :, :, :]
T1 = T1[None, None, :, :, :]
# print(I0.shape, T0.shape, T1.shape)

image = np.concatenate((I0, T0, T1), axis=1)
t_image = torch.Tensor(image)
np.save(os.path.join(MAIN_DIR, 'saved_npy', f'large_array_start'), large_array)

tuple_list=[]

print("start loop")
try:
    for i in range(0, max_t-skip, skip):
        
        if (mov_dir[i] != mov_dir[i+skip]).any():
            print("turn occurs skip")
            continue
        
        x = round(xs[i]/cell_size)
        y = round(ys[i]/cell_size)
        z = round((zs[i])/cell_size)
        #print(f"x={x}, y={y}, z={z}")
        


        shift = mov_dir[i]/np.linalg.norm(mov_dir[i])*19
        shift = np.rint(shift)
        x = int(x - shift[0])
        y = int(y - shift[1])
        print(f"after shift x={x}, y={y}, z={z}")
                
        laser_mov = mov_dir[i]/np.linalg.norm(mov_dir[i])*1e-5 # laser moves 10um in 100 timesteps

        T0 = get_T_laser(pf_args, centroids, x_laser+shift[0]/div, y_laser+shift[1]/div, z_laser, ps[i], mov_dir[i]) 
        T1 = get_T_laser(pf_args, centroids, x_laser+laser_mov[0]+shift[0]/div, y_laser+laser_mov[1]+shift[1]/div, z_laser, ps[i+skip], mov_dir[i+skip]) 
        T0 = T0.reshape((80, 80, 32), order="F")
        T1 = T1.reshape((80, 80, 32), order="F")
        t_Tr0 = torch.Tensor(T0)
        t_Tr1 = torch.Tensor(T1)
        
        tuple_list.append((x,y,z))
        first_tuple= tuple_list[0]
        last_tuple=tuple_list[-1]
        x3,y3,z3= (last_tuple[0] - first_tuple[0]), (last_tuple[1] - first_tuple[1]), (last_tuple[2] - first_tuple[2])

        print("y3=====",y3)
        x1,y1,z1= first_tuple[0],first_tuple[1],first_tuple[2],

        

        t_I0 = rve_picker(x, y, z, t_large_array)
    # print(t_I0)
        

        t_image[:, 0, :, :, :] = t_I0 /19.0 #normalization
        t_image[:, 1, :, :, :] = (t_Tr0-300) /1700.0
        t_image[:, 2, :, :, :] = (t_Tr1-300) /1700.0
        # print(t_image)
        # break
        preds = predictor(t_image)
        t_large_array = large_arr_updater(x, y, z, preds, t_large_array)

        x_cell, y_cell, z_cell= (x3+80)*cell_size, (y3+80)*cell_size,25*cell_size

        # Check if any of the values is zero and adjust accordingly
        if x3 == 0:
            x1= first_tuple[0]
            x_cell=80*cell_size
        if y3==0:
            y1= first_tuple[1]
            y_cell=80*cell_size
        x2, y2, z2 = last_tuple[0], last_tuple[1], last_tuple[2]
            
        
        if y3==0:
            melted_volume=t_large_array[x1-40:x2+40, y-40:y+40, z-28:z]
        if x3==0:
            melted_volume=t_large_array[x-40:x+40, y1-40:y2+40, z-28:z]

        l_array = melted_volume.to('cpu').detach().numpy()
        
        print("cell_size",x_cell,y_cell,z_cell)
        domain_x,domain_y,domain_z = x_cell,y_cell,z_cell
        if i % 100 == 0:
            np.save(os.path.join("/home/atwumasi/Documents/Austine/training_8November/ML_micro-main/3_predict/data/example/neper", f'process_eta'), l_array.astype(np.int8))
        print(tuple_list)

    print("done all")

except:

    import meshio

    def box_mesh_f(Nx, Ny, Nz, domain_x, domain_y, domain_z):
        """Generate a box mesh in meshio to save solutions to .vtu format
        """
        dim = 3
        x = np.linspace(0, domain_x, Nx + 1)
        y = np.linspace(0, domain_y, Ny + 1)
        z = np.linspace(0, domain_z, Nz + 1)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        points_xyz = np.stack((xv, yv, zv), axis=dim)
        points = points_xyz.reshape(-1, dim, order='F')
        points_inds = np.arange(len(points))
        points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1, order='F')
        inds1 = points_inds_xyz[:-1, :-1, :-1]
        inds2 = points_inds_xyz[1:, :-1, :-1]
        inds3 = points_inds_xyz[1:, 1:, :-1]
        inds4 = points_inds_xyz[:-1, 1:, :-1]
        inds5 = points_inds_xyz[:-1, :-1, 1:]
        inds6 = points_inds_xyz[1:, :-1, 1:]
        inds7 = points_inds_xyz[1:, 1:, 1:]
        inds8 = points_inds_xyz[:-1, 1:, 1:]
        cells = np.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), axis=dim).reshape(-1, 8, order="F")
        out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
        return out_mesh

    
    f = '/home/atwumasi/Documents/Austine/training_8November/ML_micro-main/3_predict/data/example/neper/process_eta'
    id = np.load(f+'.npy')
    Nx, Ny, Nz = np.shape(id)
    print(np.shape(id))

    # domain_x=0.0001724137931034483
    # domain_y=0.0002521551724137931
    # domain_z=5.3879310344827585e-05
    mesh = box_mesh_f(Nx=Nx, Ny=Ny, Nz=Nz, domain_x=domain_x, domain_y=domain_y, domain_z=domain_z)
    mesh.cell_data['ori_inds'] = [np.reshape(id, (-1))]
    mesh.cell_data['ori_inds'] = [id.flatten(order='F')]
    mesh.write(f+'.vtu')


# Analyze the Predicted Region for Grain Changes
import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import yaml


onp.random.seed(1)


class Field:
    """Handles polycrystal mesh, grain orientations, etc.
    """
    def __init__(self, pf_args, ori2=None):
        self.pf_args = pf_args
        self.ori2 = ori2
        self.process_neper_mesh()
        self.get_unique_ori_colors()

    def process_neper_mesh(self):
        print(f"Processing neper mesh...")
        neper_folder = os.path.join(self.pf_args['data_dir'], "neper")
        mesh = meshio.read(os.path.join(neper_folder, f"process_eta.vtu"))
        points = mesh.points
        cells = mesh.cells_dict['hexahedron']
        cell_ori_inds = onp.array(mesh.cell_data['ori_inds'][0], dtype='int64')
        print(f"cell ori inds shape {cell_ori_inds.shape}")

        # TODO: Not robust
        #Nx = round(self.pf_args['domain_x'] / points[1, 0])
        #Ny = round(self.pf_args['domain_y'] / points[Nx + 1, 1])
        #Nz = round(self.pf_args['domain_z'] / points[(Nx + 1)*(Ny + 1), 2])
        # Nx = 80
        # Ny = 117
        # Nz = 28
        assert Nx*Ny*Nz == len(cells)
        self.pf_args['Nx'] = Nx
        self.pf_args['Ny'] = Ny
        self.pf_args['Nz'] = Nz
        print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")
        print(f"Total num of finite difference cells = {len(cells)}")

        cell_points = onp.take(points, cells, axis=0)
        centroids = onp.mean(cell_points, axis=1)
        mesh_h_xyz = (self.pf_args['domain_x']/self.pf_args['Nx'], 
                      self.pf_args['domain_y']/self.pf_args['Ny'], 
                      self.pf_args['domain_z']/self.pf_args['Nz'])

        self.mesh = mesh
        self.mesh_h_xyz = mesh_h_xyz
        self.centroids = centroids
        self.cell_ori_inds = cell_ori_inds 

        pf_vtk_mesh_folder = os.path.join(self.pf_args['data_dir'], f"vtk/pf/mesh")
        os.makedirs(pf_vtk_mesh_folder, exist_ok=True)
        mesh.write(os.path.join(pf_vtk_mesh_folder, f"fd_mesh.vtu"))

        # Optionally, create a poly mesh: obj to vtu
        # file = open(os.path.join(neper_folder, "domain.obj"), 'r')
        # lines = file.readlines()
        # points = []
        # cells_inds = []
        # for i, line in enumerate(lines):
        #     l = line.split()
        #     if l[0] == 'v':
        #         points.append([float(l[1]), float(l[2]), float(l[3])])
        #     if l[0] == 'g':
        #         cells_inds.append([])
        #     if l[0] == 'f':
        #         cells_inds[-1].append([int(pt_ind) - 1 for pt_ind in l[1:]])
        # cells = [('polyhedron', cells_inds)]
        # poly_mesh = meshio.Mesh(points, cells)
        # poly_mesh.write(os.path.join(pf_vtk_mesh_folder, f"poly_mesh.vtu"))
        

    def get_unique_ori_colors(self):
        """Grain orientations and IPF colors
        """
        if self.ori2 is None:
            ori2 = Orientation.random(self.pf_args['num_oris'])
        else:
            ori2 = self.ori2

        vx = Vector3d((1, 0, 0))
        vy = Vector3d((0, 1, 0))
        vz = Vector3d((0, 0, 1))
        ipfkey_x = plot.IPFColorKeyTSL(symmetry.Oh, vx)
        rgb_x = ipfkey_x.orientation2color(ori2)
        ipfkey_y = plot.IPFColorKeyTSL(symmetry.Oh, vy)
        rgb_y = ipfkey_y.orientation2color(ori2)
        ipfkey_z = plot.IPFColorKeyTSL(symmetry.Oh, vz)
        rgb_z = ipfkey_z.orientation2color(ori2)
        rgb = onp.stack((rgb_x, rgb_y, rgb_z))

        dx = onp.array([1., 0., 0.])
        dy = onp.array([0., 1., 0.])
        dz = onp.array([0., 0., 1.])
        scipy_quat = onp.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
        r = R.from_quat(scipy_quat)
        grain_directions = onp.stack((r.apply(dx), r.apply(dy), r.apply(dz)))

        # Output orientations to numpy in the form of quaternion
        pf_numpy_folder = os.path.join(self.pf_args['data_dir'], "numpy/pf")
        os.makedirs(pf_numpy_folder, exist_ok=True)
        os.makedirs(os.path.join(pf_numpy_folder, "sols"), exist_ok=True)
        onp.save(os.path.join(pf_numpy_folder, f"quat.npy"), ori2.data)

        # Plot orientations with IPF figures
        pf_pdf_folder = os.path.join(self.pf_args['data_dir'], "pdf/pf")
        os.makedirs(pf_pdf_folder, exist_ok=True)

        # Plot IPF for those orientations
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 20,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        ori2.symmetry = symmetry.Oh
        ori2.scatter("ipf", c=rgb_x, direction=ipfkey_x.direction)
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_x.pdf'), bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_y, direction=ipfkey_y.direction)
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_y.pdf'), bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_z, direction=ipfkey_z.direction)
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_z.pdf'), bbox_inches='tight')

        # Plot the IPF legend
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 25,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        plot.IPFColorKeyTSL(symmetry.Oh).plot()
        plt.savefig(os.path.join(pf_pdf_folder, "ipf_legend.pdf"), bbox_inches='tight')

        self.unique_oris_rgb, self.unique_grain_directions = rgb, grain_directions
 

def process_eta(pf_args):
    # step = 13
    # print(pf_args)
    # print(pf)
    file_path = os.path.join(pf_args['data_dir'], "neper", "process_eta.vtu")
    mesh_w_data = meshio.read(file_path)
    cell_ori_inds = onp.array((mesh_w_data.cell_data['ori_inds'][0]), dtype='int64')

    # By default, numpy uses order='C'
    cell_ori_inds_3D = onp.reshape(cell_ori_inds, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx']))

    # This should also work
    # cell_ori_inds_3D = onp.reshape(cell_ori_inds, (pf_args['Nx'], pf_args['Ny'], pf_args['Nz']), order='F')

    print(cell_ori_inds_3D.shape)

    # T = mesh_w_data.cell_data['T'][0] ## Only for PF
    # nonliquid = T.reshape(-1) < pf_args['T_liquidus']  ## for PF 
    nonliquid = np.ones(shape=(len(cell_ori_inds)))
    # nonliquid = np.load(os.path.join(pf_args['data_dir'], "melted.npy"))
    print(f"nonliquid shape is {nonliquid.shape}")
    edges_in_order = compute_edges_in_order(pf_args)

    points = mesh_w_data.points
    cells = mesh_w_data.cells_dict['hexahedron']
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)

    domain_vol = pf_args['domain_x']*pf_args['domain_y']*pf_args['domain_z']
    volumes = domain_vol / len(cells) * onp.ones(len(cells))

    grains_combined = BFS(edges_in_order, nonliquid, cell_ori_inds, pf_args)
    print("BFS done")
    grain_vols, grain_centroids = get_aspect_ratio_inputs(grains_combined, volumes, centroids)
    eta_results = compute_aspect_ratios_and_vols(grain_vols, grain_centroids)
    # onp.save(os.path.join(pf_args['data_dir'], "eta_results.npy"), eta_results)
    # onp.save(os.path.join(pf_args['data_dir'], "neper", "eta_results.npy"), eta_results)
    #onp.save(os.path.join("/home/jyc3887", "eta_results.npy"), eta_results)

    # print(eta_results)

    return eta_results



def compute_edges_in_order(pf_args):
    Nx, Ny, Nz = pf_args['Nx'], pf_args['Ny'], pf_args['Nz']
    num_total_cells = Nx*Ny*Nz
    cell_inds = onp.arange(num_total_cells).reshape(Nz, Ny, Nx)
    edges_x = onp.stack((cell_inds[:, :, :-1], cell_inds[:, :, 1:]), axis=3).reshape(-1, 2)
    edges_y = onp.stack((cell_inds[:, :-1, :], cell_inds[:, 1:, :]), axis=3).reshape(-1, 2)
    edges_z = onp.stack((cell_inds[:-1, :, :], cell_inds[1:, :, :]), axis=3).reshape(-1, 2)
    edges = onp.vstack((edges_x, edges_y, edges_z))
    print(f"edges.shape = {edges.shape}")
    edges_in_order = [[] for _ in range(num_total_cells)]
    print(f"Re-ordering edges and face_areas...")
    for i, edge in enumerate(edges):
        node1 = edge[0]
        node2 = edge[1]
        edges_in_order[node1].append(node2)
        edges_in_order[node2].append(node1)  
    return edges_in_order


## Original BFS Implementation. Not Robust
# def BFS(edges_in_order, nonliquid, cell_ori_inds, pf_args, combined=True):
#     num_graph_nodes = len(nonliquid)
#     print(f"BFS...")
#     visited = onp.zeros(num_graph_nodes)
#     grains = [[] for _ in range(pf_args['num_oris'])]
#     for i in range(len(visited)):
#         if i % 100000 == 0:
#             print(f"i in BFS is {i}/{len(nonliquid)}")
#         if visited[i] == 0 and nonliquid[i]:
#             oris_index = cell_ori_inds[i]
#             grains[oris_index].append([])
#             queue = [i]
#             visited[i] = 1
#             while queue:
#                 s = queue.pop(0) 
#                 grains[oris_index][-1].append(s)
#                 connected_nodes = edges_in_order[s]
#                 for cn in connected_nodes:
#                     if visited[cn] == 0 and cell_ori_inds[cn] == oris_index and nonliquid[cn]:
#                         queue.append(cn)
#                         visited[cn] = 1

#     print("first loop done in BFS")

#     grains_combined = []
#     for i in range(len(grains)):
#         if i % 100000 == 0:
#             print(f"i in BFS 2 is {i}/{len(grains)}")
#         grains_oris = grains[i] 
#         for j in range(len(grains_oris)):
#             grains_combined.append(grains_oris[j])

#     if combined:
#         return grains_combined
#     else:
#         return grains


from collections import deque
import numpy as np

#Optimized BFS
def BFS(edges_in_order, nonliquid, cell_ori_inds, pf_args, combined=True):
    num_graph_nodes = len(nonliquid)
    visited = np.zeros(num_graph_nodes)
    grains = [[] for _ in range(pf_args['num_oris'])]
    for i in range(num_graph_nodes):
        if not visited[i] and nonliquid[i]:
            oris_index = cell_ori_inds[i]
            grains[oris_index].append([])
            queue = deque([i])
            visited[i] = 1
            while queue:
                s = queue.popleft()
                grains[oris_index][-1].append(s)
                for cn in edges_in_order[s]:
                    if not visited[cn] and cell_ori_inds[cn] == oris_index and nonliquid[cn]:
                        queue.append(cn)
                        visited[cn] = 1

    grains_combined = []
    for grain_list in grains:
        grains_combined.extend(grain_list)

    return grains_combined if combined else grains


def get_aspect_ratio_inputs(grains_combined, volumes, centroids):
    grain_vols = []
    grain_centroids = []
    for i in range(len(grains_combined)):
        if i % 10000 == 0:
            print(f"i in get aspect ratio inputs is {i}/{len(grains_combined)}")
        grain = grains_combined[i]
        grain_vol = onp.array([volumes[g] for g in grain])
        grain_centroid = onp.take(centroids, grain, axis=0)
        assert grain_centroid.shape == (len(grain_vol), 3)
        grain_vols.append(grain_vol)
        grain_centroids.append(grain_centroid)

    return grain_vols, grain_centroids


def compute_aspect_ratios_and_vols(grain_vols, grain_centroids):
    pca = PCA(n_components=3)
    print(f"Call compute_aspect_ratios_and_vols")
    grain_sum_vols = []
    grain_sum_aspect_ratios = []

    for i in range(len(grain_vols)):
        if i % 10000 == 0:
            print(f"i in compute aspect ratio is {i}/{len(grain_vols)}")
            
        grain_vol = grain_vols[i]
        sum_vol = onp.sum(grain_vol)
     
        if len(grain_vol) < 500:
            print(f"Grain vol too small, ignore and set aspect_ratio = 0.9")
            grain_sum_aspect_ratios.append(0.9)
        else:
            directions = grain_centroids[i]
            weighted_directions = directions * grain_vol[:, None]
            # weighted_directions = weighted_directions - onp.mean(weighted_directions, axis=0)[None, :]
            pca.fit(weighted_directions)
            components = pca.components_
            ev = pca.explained_variance_
            lengths = onp.sqrt(ev)
            aspect_ratio = 2*lengths[0]/(lengths[1] + lengths[2])
            grain_sum_aspect_ratios.append(aspect_ratio)

        grain_sum_vols.append(sum_vol)

    print(len(grain_sum_vols))
    print(len(grain_sum_aspect_ratios))
    return [grain_sum_vols, grain_sum_aspect_ratios]


def walltime(data_dir=None):
    def decorate(func):
        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            print(f"Time elapsed {time_elapsed} of function {func.__name__} on platform {platform}")
            if data_dir is not None:
                txt_dir = os.path.join(data_dir, f'txt')
                os.makedirs(txt_dir, exist_ok=True)
                with open(os.path.join(txt_dir, f"walltime_{platform}.txt"), 'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values
        return wrapper
    return decorate


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4')
    
    
def yaml_parse(yaml_filepath):
    with open(yaml_filepath) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print("YAML parameters:")
        # TODO: These are just default parameters
        print(yaml.dump(args, default_flow_style=False))
        print("These are default parameters")
    return args

    
def set_params():
    crt_file_path = os.path.dirname(__file__)
    pf_args = yaml_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    # pf_args = yaml_parse(os.path.join(crt_file_path, '3_track.yaml'))
    pf_args['case'] = "example"
    data_dir = os.path.join(crt_file_path, 'data', pf_args['case'])
    pf_args['data_dir'] = data_dir
    return pf_args

def gen_aspect_ratio():
    pf_args = set_params()
    polycrystal = Field(pf_args)
    results = process_eta(pf_args)
    grain_sum_vols, grain_sum_aspect_ratios = results
    np.save('grain_sum_aspect_ratios', grain_sum_aspect_ratios)
    np.save('grain_sum_vols', grain_sum_vols)

if __name__=="__main__":
    pf_args = set_params()
    polycrystal = Field(pf_args)
    results = process_eta(pf_args)
    grain_sum_vols, grain_sum_aspect_ratios = results
    np.save('grain_sum_aspect_ratios', grain_sum_aspect_ratios)
    np.save('grain_sum_vols', grain_sum_vols)
    # data = {'grain_sum_vols': grain_sum_vols, 
    #         'grain_sum_aspect_ratios': grain_sum_aspect_ratios}

    # # Saving the dictionary in a .npz file
    # np.savez('grain_data.npz', **data)

#mean aspect ratio

#ml_aspect_ratios= np.load('/data/proy/training_8November/ML_micro-main/3_predict/grain_sum_aspect_ratios.npy')
print(len(grain_sum_aspect_ratios))
Mean_aspect_ratio = sum(grain_sum_aspect_ratios)/len(grain_sum_aspect_ratios)
print("average_aspect_ratio=",Mean_aspect_ratio)

elapsed_time = time.time() - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
