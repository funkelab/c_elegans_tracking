import argparse
import csv
from pathlib import Path

import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
import pandas.errors
import tifffile
import toml
import zarr
import zarr.creation
from funlib.geometry import Coordinate, Roi
from motile_tracker.data_model import Tracks
from tqdm import tqdm

DIR_TEMPLATE = "Decon_reg_{time}"
STRAIGHT_FILE_TEMPLATE = "Decon_reg_{time}_straight.tif"
SEG_FILE_TEMPLATE = "Decon_reg_{time}_masks.tif"
TWISTED_FILE_TEMPLATE = "Decon_reg_{time}.tif"


# a left and right padding for each of the three dimensions (z, y, x)
def get_center_pad_widths(
    arr_shapes: list[Coordinate], target_shape: Coordinate
) -> list[tuple[Coordinate, Coordinate]]:
    """Gets the pad widths necessary to center each array in arrs in a new array of the
    given shape

    Args:
        arrs (list[tuple]): A list of current array shapes to pad to the given shape by
            placing them in the center.
        shape (tuple): The desired shape of each numpy array. Should be >= the shape of
            each array (no cropping implemented)
    Returns:
        list[PadWidth]: A list of pad widths to be used in np.pad, each corresponding
        to one array in arrs. Each pad width is a tuple of three elements,
    """
    pad_widths: list[tuple[Coordinate, Coordinate]] = []
    for curr_shape in arr_shapes:
        pad_amt = target_shape - curr_shape
        half_pad_amt = pad_amt // 2
        pad_widths.append((half_pad_amt, pad_amt - half_pad_amt))
    return pad_widths


def _get_store_padding(store) -> tuple[Coordinate, list[tuple[Coordinate, Coordinate]]]:
    """_summary_

    Args:
        store (_type_): _description_

    Returns:
        tuple[Coordinate, list[tuple[Coordinate, Coordinate]]]: The shape of one frame,
            and the pad amounts for each frame
    """
    _test_exists(store)
    store = zarr.open(store, "r")
    output_frame_shape = store.shape[1:]
    pad_widths = store.attrs["pad_widths"]
    return output_frame_shape, pad_widths


def convert_raw_straightened(
    raw_path: Path,
    output_store: Path,
    time_range=(11, 85),
    alignment: str = "center",  # center, match_store, (soon will have seam cell option)
    store_to_match: Path | None = None,
):
    # get the shapes of all arrays
    shapes: list[Coordinate] = []
    for i in tqdm(
        range(*time_range), desc="Getting shapes of each straightened time point"
    ):
        file = (
            raw_path / DIR_TEMPLATE.format(time=i) / STRAIGHT_FILE_TEMPLATE.format(time=i)
        )
        assert file.is_file(), f"File {file} does not exist"
        tif = tifffile.TiffFile(file)
        num_z_slices = len(tif.pages)
        yx_shape = tif.pages[0].shape
        shape = (num_z_slices, *yx_shape)
        shapes.append(Coordinate(shape))

    output_frame_shape: Coordinate
    pad_widths: list[tuple[Coordinate, Coordinate]]
    if alignment == "match_store":
        assert (
            store_to_match is not None
        ), "Must provide store to match size and pad_widths"
        output_frame_shape, pad_widths = _get_store_padding(store_to_match)
    elif alignment == "center":
        # determine the maximum extent in each dimension
        ndim = len(shapes[0])
        output_frame_shape = Coordinate(
            max([shape[dim] for shape in shapes]) for dim in range(ndim)
        )
        # figure out the padding for each time point
        pad_widths = get_center_pad_widths(shapes, output_frame_shape)
    else:
        raise ValueError(
            f"Alignment {alignment} not in valid options: ['center','match_store']"
        )
    # prepare zarr
    num_times = time_range[1] - time_range[0]
    ds_shape = (num_times, *output_frame_shape)
    offset = (time_range[0], 0, 0, 0)
    voxel_size = (1, 1, 1, 1)
    axis_names = ("t", "z", "y", "x")
    dtype = np.uint16  # the tiffs are float 32. However, the floating points are not used
    # and the max value is 63430. So we will save as uint16 to be efficient

    target_array = fp.prepare_ds(
        store=output_store,
        shape=ds_shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        dtype=dtype,
        mode="w",
    )

    # add pad width list for retrieval later (to be replaced with custom_metadata)
    zarr_array = zarr.open(output_store, mode="a")
    zarr_array.attrs["pad_widths"] = pad_widths

    # load the actual data, pad, and write to zarr
    for i in tqdm(
        range(*time_range),
        desc="Loading, padding, and saving each straightened time point",
    ):
        file = (
            raw_path / DIR_TEMPLATE.format(time=i) / STRAIGHT_FILE_TEMPLATE.format(time=i)
        )
        assert file.is_file(), f"File {file} does not exist"
        arr = tifffile.imread(file)
        time_idx = i - time_range[0]
        padding = pad_widths[time_idx]

        # do it properly with funlib persistence
        roi = Roi(offset=(i, *padding[0]), shape=(1, *arr.shape))
        arr = np.expand_dims(arr, axis=0)
        target_array[roi] = arr


def convert_raw_twisted(
    raw_path: Path,
    output_store: Path,
    time_range=(11, 85),
):
    # get the shapes of one frame

    file = raw_path / TWISTED_FILE_TEMPLATE.format(time=time_range[0])
    assert file.is_file(), f"File {file} does not exist"
    tif = tifffile.TiffFile(file)
    num_z_slices = len(tif.pages)
    yx_shape = tif.pages[0].shape
    output_frame_shape = (num_z_slices, *yx_shape)
    # pad_widths = None

    # prepare zarr
    num_times = time_range[1] - time_range[0]
    ds_shape = (num_times, *output_frame_shape)
    offset = (time_range[0], 0, 0, 0)
    voxel_size = (1, 1, 1, 1)
    axis_names = ("t", "z", "y", "x")
    dtype = np.uint16
    # the tiffs are float 32. However, the floating points are not used
    # and the max value is 63430. So we will save as uint16 to be efficient
    target_array = fp.prepare_ds(
        store=output_store,
        shape=ds_shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        dtype=dtype,
        mode="w",
    )

    # add pad width list for retrieval later (to be replaced with custom_metadata)
    # zarr_array = zarr.open(output_store, mode='a')
    # zarr_array.attrs["pad_widths"] = pad_widths

    # load the actual data, pad, and write to zarr
    for i in tqdm(range(*time_range), desc="Loading and saving each twisted time point"):
        file = raw_path / TWISTED_FILE_TEMPLATE.format(time=i)
        assert file.is_file(), f"File {file} does not exist"
        arr = tifffile.imread(file)
        # time_idx = i - time_range[0]
        # padding = pad_widths[time_idx]

        # do it properly with funlib persistence
        roi = Roi(offset=(i, 0, 0, 0), shape=(1, *arr.shape))
        arr = np.expand_dims(arr, axis=0)
        target_array[roi] = arr


def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


def convert_tracks(
    annotations_path: Path,
    path_after_time: str,
    output_path: Path,
    time_range: tuple[int, int],
    offsets: list[Coordinate] | None,
):
    graph = nx.DiGraph()
    node_id = 0
    name_to_id = {}
    for i in tqdm(range(*time_range)):
        offset = offsets[i - time_range[0]] if offsets is not None else [0, 0, 0]
        file = annotations_path / DIR_TEMPLATE.format(time=i) / path_after_time
        _test_exists(file)
        try:
            df = pd.read_csv(file)
        except pandas.errors.EmptyDataError as e:
            raise ValueError(f"{file} cannot be read by pandas - might be empty") from e
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
                "pixel_loc": [row["z_voxels"], row["y_voxels"], row["x_voxels"]],
                "name": row["name"],
            }
            graph.add_node(node_id, **attrs)
            if name in name_to_id:
                graph.add_edge(name_to_id[name], node_id)
            name_to_id[name] = node_id
            node_id += 1

    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )
    tracks = Tracks(graph, ndim=4)
    tracks.save(output_path)


def convert_lattice_points(
    annotations_path: Path,
    path_after_time: str,
    output_path: Path,
    time_range: tuple[int, int],
    offsets: list[Coordinate] | None,
):
    array = np.zeros(shape=(time_range[1] - time_range[0], 11, 2, 3))
    names = ["a0", "h0", "h1", "h2", "v1", "v2", "v3", "v4", "v5", "v6", "t"]
    sides = ["r", "l"]
    axis_names = ["z", "y", "x"]

    for i in tqdm(range(*time_range)):
        offset = offsets[i - time_range[0]] if offsets is not None else [0, 0, 0]
        file = annotations_path / DIR_TEMPLATE.format(time=i) / path_after_time
        _test_exists(file)
        df = pd.read_csv(file)
        time = i - time_range[0]
        z_offset, y_offset, x_offset = offset
        df["z_voxels"] += z_offset
        df["y_voxels"] += y_offset
        df["x_voxels"] += x_offset
        for _, row in df.iterrows():
            pos = [row["z_voxels"], row["y_voxels"], row["x_voxels"]]
            name = row["name"].lower()
            seam_cell_name = name[0:-1]
            if seam_cell_name not in names:
                continue  # this is a virtual seam cell, skip it for now
            seam_cell_idx = names.index(seam_cell_name)
            side = name[-1]
            side_idx = sides.index(side)
            array[time, seam_cell_idx, side_idx, :] = pos

    metadata = {"lattice_point_names": names, "sides": sides, "axis_names": axis_names}
    zarray = zarr.creation.array(store=output_path, data=array, overwrite=True)
    zarray.attrs.update(metadata)


def convert_points(
    annotations_path: Path,
    path_after_time: str,
    output_file: Path,
    time_range: tuple[int, int],
    offsets: list[Coordinate],
):
    with open(output_file, "w") as f:
        header = ["id", "t", "z", "y", "x", "name"]
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()

        node_id = 0
        for i in tqdm(range(*time_range)):
            offset = offsets[i - time_range[0]] if offsets is not None else [0, 0, 0]
            file = annotations_path / DIR_TEMPLATE.format(time=i) / path_after_time
            _test_exists(file)

            df = pd.read_csv(file)
            df["t"] = i - time_range[0]
            z_offset = offset[0]
            y_offset = offset[1]
            x_offset = offset[2]
            df["z"] = df["z_voxels"] + z_offset
            df["y"] = df["y_voxels"] + y_offset
            df["x"] = df["x_voxels"] + x_offset
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["id"] = node_id
                writer.writerow(row_dict)
                node_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--seam-cell-raw", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--seam-cell-tracks", action="store_true")
    parser.add_argument("--lattice-points", action="store_true")
    parser.add_argument("--seg-centers", action="store_true")
    args = parser.parse_args()
    config = toml.load(args.config)
    if args.time_range is not None:
        config["conversion"]["time_range"] = args.time_range

    # verify input locations
    input_config = config["input"]
    data_base_path = Path(input_config["base_path"])
    _test_exists(data_base_path)
    raw_path = data_base_path / input_config["raw_path"]
    _test_exists(raw_path)

    # create output location
    output_config = config["output"]
    output_base_path = Path(output_config["base_path"])
    output_base_path.mkdir(exist_ok=True, parents=True)
    output_zarr = output_base_path / output_config["zarr"]
    raw_store = output_zarr / output_config["raw_group"]

    # get conversion parameters
    conv_config = config["conversion"]
    straightened = conv_config["straightened"]
    if straightened:
        alignment = conv_config["alignment"]
    else:
        alignment = None
    time_range = conv_config["time_range"]
    print(time_range)

    # run conversion
    if args.raw:
        if straightened:
            convert_raw_straightened(
                raw_path, raw_store, time_range=time_range, alignment=alignment
            )
        else:
            convert_raw_twisted(raw_path, raw_store, time_range=time_range)
    if args.seam_cell_raw:
        seam_cell_raw_path = data_base_path / input_config["seam_cell_path"]
        _test_exists(seam_cell_raw_path)
        seam_cell_raw_store = output_zarr / output_config["seam_cell_group"]
        if straightened:
            convert_raw_straightened(
                seam_cell_raw_path,
                seam_cell_raw_store,
                time_range=time_range,
                alignment="match_store",
                store_to_match=raw_store,
            )
        else:
            convert_raw_twisted(
                seam_cell_raw_path, seam_cell_raw_store, time_range=time_range
            )
    if args.seg:
        seg_path = data_base_path / input_config["seg_path"]
        _test_exists(seg_path)
        seg_store = output_zarr / output_config["seg_group"]
        if straightened:
            convert_raw_straightened(
                seg_path,
                seg_store,
                time_range=time_range,
                alignment="match_store",
                store_to_match=raw_store,
            )
        else:
            convert_raw_twisted(seg_path, seg_store, time_range=time_range)
    if args.manual:
        manual_base_path = data_base_path / input_config["manual_tracks_base"]
        _test_exists(manual_base_path)
        manual_end_path = input_config["manual_tracks_end"]
        manual_output_path: Path = (
            output_zarr / output_config["manual_tracks_dir"]
        )  # put it inside the zarr
        manual_output_path.mkdir(exist_ok=True, parents=False)
        if straightened:
            _, pad_widths = _get_store_padding(raw_store)
            offsets = [pad_width[0] for pad_width in pad_widths]
        else:
            offsets = None
        convert_tracks(
            manual_base_path,
            manual_end_path,
            manual_output_path,
            time_range=time_range,
            offsets=offsets,
        )
    if args.seam_cell_tracks:
        seam_cell_base_path = data_base_path / input_config["seam_cell_tracks_base"]
        _test_exists(seam_cell_base_path)
        seam_cell_end_path = input_config["seam_cell_tracks_end"]
        seam_cell_output_path: Path = (
            output_zarr / output_config["seam_cell_tracks_dir"]
        )  # put it inside the zarr
        seam_cell_output_path.mkdir(exist_ok=True, parents=False)
        if straightened:
            _, pad_widths = _get_store_padding(raw_store)
            offsets = [pad_width[0] for pad_width in pad_widths]
        else:
            offsets = None
        convert_tracks(
            seam_cell_base_path,
            seam_cell_end_path,
            seam_cell_output_path,
            time_range=time_range,
            offsets=offsets,
        )
    if args.lattice_points:
        seam_cell_base_path = data_base_path / input_config["seam_cell_tracks_base"]
        _test_exists(seam_cell_base_path)
        seam_cell_end_path = input_config["seam_cell_tracks_end"]
        lattice_point_output_path: Path = (
            output_zarr / output_config["lattice_points_dir"]
        )  # put it inside the zarr
        lattice_point_output_path.mkdir(exist_ok=True, parents=False)
        if straightened:
            _, pad_widths = _get_store_padding(raw_store)
            offsets = [pad_width[0] for pad_width in pad_widths]
        else:
            offsets = None
        convert_lattice_points(
            seam_cell_base_path,
            seam_cell_end_path,
            lattice_point_output_path,
            time_range=time_range,
            offsets=offsets,
        )
    if args.seg_centers:
        seg_centers_base_path = data_base_path / input_config["seg_centers_base"]
        _test_exists(seg_centers_base_path)
        seg_centers_end_path = input_config["seg_centers_end"]
        seg_centers_output_path: Path = (
            output_zarr / output_config["seg_centers_file"]
        )  # put it inside the zarr
        if straightened:
            _, pad_widths = _get_store_padding(raw_store)
            offsets = [pad_width[0] for pad_width in pad_widths]
        else:
            offsets = None
        convert_points(
            seg_centers_base_path,
            seg_centers_end_path,
            seg_centers_output_path,
            time_range=time_range,
            offsets=offsets,
        )
