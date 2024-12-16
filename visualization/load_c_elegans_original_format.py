from pathlib import Path
import tifffile
import numpy as np
import napari
import pandas as pd
from tqdm import tqdm
import networkx as nx
from motile_plugin.data_model import SolutionTracks
from motile_plugin.application_menus import MainApp
from motile_plugin.data_views.views_coordinator.tracks_viewer import TracksViewer
import argparse


DIR_TEMPLATE = "Decon_reg_{time}"
STRAIGHT_FILE_TEMPLATE = "Decon_reg_{time}_straight.tif"
SEG_FILE_TEMPLATE = "Decon_reg_{time}_masks.tif"
TWISTED_FILE_TEMPLATE = "Decon_reg_{time}.tif"

def top_left_pad(arrs, shape):
    ndim = len(shape)
    for i, arr in enumerate(arrs):
        pad_amt = tuple(shape[dim] - arr.shape[dim] for dim in range(ndim))
        pad_width = tuple((0, pad_amt[dim]) for dim in range(ndim))
        padded = np.pad(arr, pad_width)
        arrs[i] = padded

def center_pad(arrs, shape):
    ndim = len(shape)
    offsets = []
    for i, arr in enumerate(arrs):
        pad_amt = tuple(shape[dim] - arr.shape[dim] for dim in range(ndim))
        half_pad_amt = tuple(pad_amt[dim] // 2 for dim in range(ndim))
        pad_width = tuple((half_pad_amt[dim], pad_amt[dim] - half_pad_amt[dim]) for dim in range(ndim))
        offsets.append(half_pad_amt)
        padded = np.pad(arr, pad_width)
        arrs[i] = padded
    
    return offsets


def pad_to_seam_cell(arrs, seam_cell_locs):
    ndim = len(arrs[0].shape)

    max_loc = tuple(max(seam_cell_locs[:, dim]) for dim in range(ndim))
    print(f"{max_loc=}")
    offsets = [tuple(int(max_loc[dim] - seam_cell_loc[dim]) for dim in range(ndim)) for seam_cell_loc in seam_cell_locs]
    print(f"{offsets=}")
    shapes = []
    for arr, offset in zip(arrs, offsets, strict=True):
        shapes.append(tuple(arr.shape[dim] + offset[dim] for dim in range(ndim)))
    max_shape = tuple(max([shape[dim] for shape in shapes]) for dim in range(ndim))
    print(f"{max_shape=}")
    pad_widths = []
    i = 0
    for arr, offset in zip(arrs, offsets, strict=True):
        upper_pad_amt = tuple(max_shape[dim] - arr.shape[dim] - offset[dim] for dim in range(ndim))
        pad_width = tuple((offset[dim], upper_pad_amt[dim]) for dim in range(ndim))
        print(f"{pad_width=}")
        pad_widths.append(pad_width)
        padded = np.pad(arr, pad_width)
        arrs[i] = padded
        i += 1
    
    return offsets

def load_raw(raw_path: Path, time_range=(11, 85), straightened=True, seam_cells=None):
    arrs = []
    shapes = []
    offsets = []
    for i in tqdm(range(*time_range)):
        if straightened:
            file = raw_path / DIR_TEMPLATE.format(time=i) / STRAIGHT_FILE_TEMPLATE.format(time=i)
        else:
            file = raw_path / TWISTED_FILE_TEMPLATE.format(time=i)
        
        assert file.is_file(), f"File {file} does not exist"
        arrs.append(tifffile.imread(file))
        shapes.append(arrs[i - time_range[0]].shape)
    print(f"shapes: {shapes}")
    ndim = len(shapes[0])
    max_shape = tuple(max([shape[dim] for shape in shapes]) for dim in range(ndim))
    if seam_cells is None:
        offsets = center_pad(arrs, max_shape)
    else:
        offsets = pad_to_seam_cell(arrs, seam_cells)

    
    return np.array(arrs), offsets

def load_seg(seg_path: Path, time_range=(11, 85)):
    arrs = []
    shapes = []
    offsets = []
    for i in tqdm(range(*time_range)):
        file = seg_path / SEG_FILE_TEMPLATE.format(time=i)
        
        assert file.is_file(), f"File {file} does not exist"
        arrs.append(tifffile.imread(file))
        shapes.append(arrs[i - time_range[0]].shape)
    ndim = len(shapes[0])
    max_shape = tuple(max([shape[dim] for shape in shapes]) for dim in range(ndim))
    offsets = center_pad(arrs, max_shape)
    
    return np.array(arrs, dtype=np.uint64), offsets

def load_points(annotations_path, time_range, offsets, straightened=True):
    points = []
    for i in tqdm(range(*time_range)):
        offset = offsets[i - time_range[0]]
        if straightened:
            file = annotations_path / DIR_TEMPLATE.format(time=i) / "straightened_annotations" / "straightened_annotations.csv"
        else:
            file = annotations_path / DIR_TEMPLATE.format(time=i) / "integrated_annotation" / "annotations.csv"
        
        assert file.is_file(), f"File {file} does not exist"
        df = pd.read_csv(file)
        df["time"] = i - time_range[0]
        z_offset = offset[0]
        y_offset = offset[1]
        x_offset = offset[2]
        df["z_voxels"] += z_offset
        df["y_voxels"] += y_offset
        df["x_voxels"] += x_offset
        coords_only = df[["time", "z_voxels", "y_voxels", "x_voxels"]]
        points_in_frame = coords_only.to_numpy()
        points.append(points_in_frame)
    
    points = np.vstack(points)
    return points

def load_graph(annotations_path, path_after_time, time_range, offsets) :
    graph = nx.DiGraph()
    node_id = 0
    name_to_id = {}
    for i in tqdm(range(*time_range)):
        offset = offsets[i - time_range[0]] if offsets is not None else [0, 0, 0]
        if straightened:
            file = annotations_path / DIR_TEMPLATE.format(time=i) / path_after_time
            # file = annotations_path / DIR_TEMPLATE.format(time=i) / "straightened_annotations" / "straightened_annotations.csv"
        else:
            file = annotations_path / DIR_TEMPLATE.format(time=i) / "integrated_annotation" / "annotations.csv"
        assert file.is_file(), f"File {file} does not exist"
        df = pd.read_csv(file)
        df["time"] = i - time_range[0]
        z_offset = offset[0]
        y_offset = offset[1]
        x_offset = offset[2]
        df["z_voxels"] += z_offset
        df["y_voxels"] += y_offset
        df["x_voxels"] += x_offset
        for _, row in df.iterrows():
            name = row["name"]
            attrs = {
                "time": row["time"],
                "pos": [row["z_voxels"], row["y_voxels"], row["x_voxels"]],
                "name": row["name"]
            }
            graph.add_node(node_id, **attrs)
            if name in name_to_id:
                graph.add_edge(name_to_id[name], node_id)
            name_to_id[name] = node_id
            node_id += 1
    
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def load_straightened(time_range):

    base_path = Path("/Volumes/funke/data/ForCarolineFromRyan")
    
    manual_annotations_path = base_path / "Annotations" / "Straightened" / "Hand_Annotated"
    carsen_annotations_path = base_path / "Annotations" / "Straightened" / "Carsen_Annotations"
    assert manual_annotations_path.is_dir(), f"Dir {manual_annotations_path} does not exist"
    raw, offsets = load_raw(raw_path, time_range)
    # seam_cell_raw, seam_cell_offsets = load_raw(seam_cell_raw_path, time_range)
    # assert offsets == seam_cell_offsets, f"Offsets from RegB {offsets} and RegA {seam_cell_offsets} are not equal"
    manual_graph = load_graph(manual_annotations_path, time_range, offsets)
    carsen_points = load_points(carsen_annotations_path, time_range, offsets)
    return raw, manual_graph, carsen_points


def _test_exists(path):
    assert path.exists(), f"{path} does not exist"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twisted", action="store_true")
    parser.add_argument("--time-range", nargs=2, type=int, default=(11, 85))
    parser.add_argument("--seam-cells", action="store_true")
    args = parser.parse_args()
    time_range = args.time_range
    straightened = not args.twisted
    
    base_path = Path("/Volumes/funke/data/ForCarolineFromRyan")
    dirname = "Straightened" if straightened else "Twisted"

    # set up napari
    viewer = napari.Viewer()
    motile_widget = MainApp(viewer)
    viewer.window.add_dock_widget(motile_widget)

    # load the seam cells to do registration
    seam_cell_path = base_path / "Seam_Cells" / dirname
    _test_exists(seam_cell_path)
    if straightened:
        path_after_time = "straightened_seamcells/straightened_seamcells.csv"
    else:
        path_after_time = "seam_cell_final/seam_cells.csv"
    seam_cells = load_graph(seam_cell_path, path_after_time, time_range, offsets=None)
    target_seam_cells = []
    target = "H1R"
    for node, data in seam_cells.nodes(data=True):
        if data["name"] == target:
            target_seam_cells.append([data["time"], *data["pos"]])
    target_seam_cells = np.array(sorted(target_seam_cells, key=lambda x: x[0]))[:, 1:]

    raw_path = base_path / "RegB" / dirname
    _test_exists(raw_path)
    raw_aligned, offsets = load_raw(raw_path, time_range, straightened=straightened, seam_cells=target_seam_cells)
    viewer.add_image(raw_aligned, name="RegB_aligned")
    # raw, offsets = load_raw(raw_path, time_range, straightened=straightened)
    # viewer.add_image(raw, name="RegB")

    seg_path = base_path / "deep_learning"
    assert seg_path.is_dir(), f"Dir {seg_path} does not exist"   
    seg, seg_offsets = load_seg(seg_path, time_range=time_range)
    viewer.add_labels(seg, name="cellpose_seg")

    carsen_annotations_path = base_path / "Annotations" / dirname / "Carsen_Annotations"
    _test_exists(carsen_annotations_path)
    carsen_points = load_points(carsen_annotations_path, time_range, offsets, straightened=straightened)
    viewer.add_points(data=carsen_points, name="carsen_annotations", face_color="pink", size=5)


    manual_annotations_path = base_path / "Annotations" / dirname / "Hand_Annotated"
    _test_exists(manual_annotations_path)
    if straightened:
        path_after_time = "straightened_annotations/straightened_annotations.csv"
    else:
        path_after_time = "integrated_annotations/annotations.csv"
    manual_graph = load_graph(manual_annotations_path, path_after_time, time_range, offsets)   
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.tracks_list.add_tracks(SolutionTracks(manual_graph, ndim=4), "manual_annotations")

    if args.seam_cells:
        seam_cell_raw_path = base_path / "RegA" / dirname
        _test_exists(seam_cell_raw_path)
        seam_cell_raw, seam_cell_offsets = load_raw(seam_cell_raw_path, time_range, straightened=straightened, seam_cells=target_seam_cells)
        assert offsets == seam_cell_offsets, f"Offsets from RegB {offsets} and RegA {seam_cell_offsets} are not equal"
        viewer.add_image(seam_cell_raw, name="RegA")
        
        tracks_viewer.tracks_list.add_tracks(SolutionTracks(seam_cells, ndim=4), "seam_cells")

    napari.run()