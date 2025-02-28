import openeo
import numpy as np

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [4.14697, 4.92183],
            [4.14697, 4.68837],
            [4.49853, 4.68837],
            [4.49853, 4.92183],
            [4.14697, 4.92183],
        ]
    ],
}

sentinel1_cube = connection.load_collection(
    "SENTINEL1_GRD",
    spatial_extent={"west": -94.5, "south": 29, "east": -94, "north": 29.5},
    temporal_extent=["2022-02-03", "2022-02-08"],
    bands=["VV"]
)
sentinel1_cube = sentinel1_cube.sar_backscatter(coefficient="sigma0-ellipsoid")

sentinel1_cube = sentinel1_cube.apply(process=lambda data: 10 * openeo.processes.log(data, base=10))

filter_window = np.ones([601, 601])
factor = 1 / np.prod(filter_window.shape)

thresholds = sentinel1_cube.apply_kernel(kernel=filter_window, factor=factor)

threshold_shift = 3.5
thresholds = thresholds - threshold_shift
thresholds = thresholds.rename_labels(dimension="bands", target=["threshold"], source=["VV"])
sentinel1_cube= sentinel1_cube.rename_labels(dimension="bands", target=["amplitude"], source=["VV"])

merged_cube = sentinel1_cube.merge_cubes(thresholds)
oil_spill = merged_cube.band("amplitude") < merged_cube.band("threshold")
oil_spill = oil_spill.max_time()

"""process_ids = [process["id"] for process in connection.list_processes()]
process_id = process_ids[:30]
print("process_id", process_id)
print(merged_cube.band("amplitude"))"""

oil_spill.execute_batch(outputfile="oil_spill.nc", out_format="netCDF")
#oil_spill.download("oil_spill.tiff")