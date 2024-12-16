from pathlib import Path
import tifffile
import numpy as np
from tqdm import tqdm
import zarr
from funlib.geometry import Roi, Coordinate
import funlib.persistence as fp
import argparse


DIR_TEMPLATE = "Decon_reg_{time}"
STRAIGHT_FILE_TEMPLATE = "Decon_reg_{time}_straight.tif"
SEG_FILE_TEMPLATE = "Decon_reg_{time}_masks.tif"
TWISTED_FILE_TEMPLATE = "Decon_reg_{time}.tif"

# def pad_to_seam_cell(arr_shapes: list[Coordinate], seam_cell_locs: np.ndarray)-> list[tuple[Coordinate, Coordinate]]:
#     ndim = arr_shapes[0].dims
#     # find max seam cell location to move everything to
#     max_loc = Coordinate(max(seam_cell_locs[:, dim]) for dim in range(ndim))
#     print(f"{max_loc=}")
#     # find offset needed for each frame to move seam cell to that location
#     offsets = [max_loc - Coordinate(seam_cell_loc) for seam_cell_loc in seam_cell_locs]
#     print(f"{offsets=}")
#     max_shape = np.max(np.array([shape + offset for shape, offset in zip(arr_shapes, offsets, strict=True)]), axis=1)
#     print(f"{max_shape=}")
#     pad_widths = []
#     for shape, offset in zip(arr_shapes, offsets, strict=True):
#         upper_pad_amt = max_shape - shape - offset
#         pad_width = (offset, upper_pad_amt)
#         print(f"{pad_width=}")
#         pad_widths.append(pad_width)
#     return pad_widths
    
# a left and right padding for each of the three dimensions (z, y, x)
def get_center_pad_widths(arr_shapes: list[Coordinate], target_shape: Coordinate) -> list[tuple[Coordinate, Coordinate]]:
    """Gets the pad widths necessary to center each array in arrs in a new array of the given shape

    Args:
        arrs (list[tuple]): A list of current array shapes to pad to the given shape by placing
            them in the center.
        shape (tuple): The desired shape of each numpy array. Should be >= the shape of each
            array (no cropping implemented)
    Returns:
        list[PadWidth]: A list of pad widths to be used in np.pad, each corresponding
        to one array in arrs. Each pad width is a tuple of three elements, 
    """
    pad_widths : list[tuple[Coordinate, Coordinate]]= []
    for curr_shape in arr_shapes:
        pad_amt = target_shape - curr_shape
        half_pad_amt = pad_amt // 2
        pad_widths.append((half_pad_amt, pad_amt - half_pad_amt))
    return pad_widths

def convert_raw_straigtened(
        raw_path: Path,
        output_zarr: Path,
        output_group: str,
        time_range=(11, 85)
    ):
    # get the shapes of all arrays 
    shapes : list[Coordinate] = []
    for i in tqdm(range(*time_range), desc="Getting shapes of each straightened time point"):
        file = raw_path / DIR_TEMPLATE.format(time=i) / STRAIGHT_FILE_TEMPLATE.format(time=i)
        assert file.is_file(), f"File {file} does not exist"
        tif = tifffile.TiffFile(file)
        num_z_slices = len(tif.pages)
        yx_shape = tif.pages[0].shape
        shape = (num_z_slices, *yx_shape)
        shapes.append(Coordinate(shape))

    # determine the maximum extent in each dimension
    ndim = len(shapes[0])
    max_shape = Coordinate(max([shape[dim] for shape in shapes]) for dim in range(ndim))

    # figure out the padding for each time point
    pad_widths = get_center_pad_widths(shapes, max_shape)


    # prepare zarr
    num_times = time_range[1] - time_range[0]
    ds_shape = (num_times, *max_shape)
    offset = (time_range[0], 0, 0, 0)
    voxel_size = (1, 1, 1, 1)
    axis_names = ("t", "z", "y", "x")   
    dtype = np.uint16  # the tiffs are float 32. However, the floating points are not used
    # and the max value is 63430. So we will save as uint16 to be efficient
    
    fp.prepare_ds(
        store=output_zarr,
        path=output_group,
        shape=ds_shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        dtype=dtype,
    )
    
    # add pad width list for retrieval later
    zarr_array = zarr.open(output_zarr, path=output_group, mode='a')
    zarr_array.attrs["pad_widths"] = pad_widths
    print("zarr array dtype ", zarr_array.dtype)

    # load the actual data, pad, and write to zarr
    # target_array : fp.Array= fp.open_ds(store=output_zarr, path=output_group, mode='a')
    for i in tqdm(range(*time_range), desc="Loading, padding, and saving each straightened time point"):
        file = raw_path / DIR_TEMPLATE.format(time=i) / STRAIGHT_FILE_TEMPLATE.format(time=i)
        assert file.is_file(), f"File {file} does not exist"
        print("starting reading")
        arr = tifffile.imread(file)
        print("done reading")
        padding = pad_widths[i - time_range[0]]
        # roi = Roi(offset=(i, *padding[0]), shape=(1, *arr.shape))
        roi = Roi(offset=padding[0], shape=arr.shape)
        arr = np.expand_dims(arr, axis=0)
        # slices = target_array._Array__slices(roi)[1:]
        # print(roi, target_array._Array__slices(roi))
        print("starting assignment")
        # zarr_array[slices] = arr
        zarr_array[i - time_range[0]][roi.to_slices()] = arr[0]
        print("done assignment")

def _test_exists(path):
    assert path.exists(), f"{path} does not exist"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--twisted", action="store_true")
    parser.add_argument("--time-range", nargs=2, type=int, default=(11, 85))
    args = parser.parse_args()
    data_base_path = Path("/Volumes/funke/data/ForCarolineFromRyan")
    _test_exists(data_base_path)
    output_base_path =  Path("/Volumes/funke/data/lightsheet/shroff_c_elegans/post_twitching_neurons")
    output_base_path.mkdir(exist_ok=True, parents=True)
    straigtened_output_zarr = output_base_path / "straightened.zarr"

    straightened_raw_path = data_base_path / "RegB" / "Straightened"
    _test_exists(straightened_raw_path)
    straightened_seam_cell_raw_path = data_base_path / "RegA" / "Straightened"
    _test_exists(straightened_seam_cell_raw_path)

    convert_raw_straigtened(straightened_raw_path, straigtened_output_zarr, "RegB", time_range=args.time_range)
