from osgeo import gdal
import tkinter as tk
from tkinter import filedialog, simpledialog
import os

def create_vrt_and_warp(MYD021KM_path, MYD03_path, band_index = 1, output_tif='ouput.tif', swath_type_indicator = 1):
    """Writes and executes a vrt file to correct hdf warping.\n
    Produces a new .tif file\n
    swath_type_indicator:\n
    1: 1km refSB\n
    2: 500m refSB aggrigate\n
    3: 250 refSb aggrigate\n
    4: 1km Emissive"""

    # check that gdal can reach all hdf datasets
    # unsure if this is actually required
    gdal.AllRegister()

    vrt_path = 'temp_modis.vrt'
    swath_type = "EV_1KM_RefSB"

    # associate swath type indicator with swath type
    # note: defaults to emissive if out of bounds input given
    # this was previously a match case statement, but this did not work for some reason.
    if swath_type_indicator == 1:
        swath_type = "EV_1KM_RefSB"
    elif swath_type_indicator == 2:
        swath_type = "EV_500_Aggr1km_RefSB"
    elif swath_type_indicator == 3:
        swath_type = "EV_250_Aggr1km_RefSB"
    else:
        swath_type = "EV_1KM_Emissive"


    # build full paths to hdf datasets
    data_subdataset = f'HDF4_EOS:EOS_SWATH:"{MYD021KM_path}":MODIS_SWATH_Type_L1B:{swath_type}'
    print(data_subdataset)
    lat_subdataset = f'HDF4_EOS:EOS_SWATH:"{MYD03_path}":MODIS_Swath_Type_GEO:Latitude'
    lon_subdataset  = f'HDF4_EOS:EOS_SWATH:"{MYD03_path}":MODIS_Swath_Type_GEO:Longitude'

    # open to get size and datatype
    ds = gdal.Open(data_subdataset)
    if ds is None:
        raise RuntimeError("Failed to open data subdataset")
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    dtype = gdal.GetDataTypeName(ds.GetRasterBand(band_index).DataType)

    # create the vrt file content
    vrt_content = f'''<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
  <SRS>WGS84</SRS>
  <GeoTransform>0,1,0,0,0,-1</GeoTransform>
  <Metadata domain="GEOLOCATION">
    <MDI key="X_DATASET">{lon_subdataset}</MDI>
    <MDI key="X_BAND">1</MDI>
    <MDI key="Y_DATASET">{lat_subdataset}</MDI>
    <MDI key="Y_BAND">1</MDI>
    <MDI key="PIXEL_OFFSET">0</MDI>
    <MDI key="LINE_OFFSET">0</MDI>
    <MDI key="PIXEL_STEP">1</MDI>
    <MDI key="LINE_STEP">1</MDI>
  </Metadata>

  <VRTRasterBand dataType="{dtype}" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="0">{data_subdataset}</SourceFilename>
      <SourceBand>{band_index}</SourceBand>
      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="{dtype}" BlockXSize="{xsize}" BlockYSize="1"/>
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
    </SimpleSource>
    <ColorInterp>Gray</ColorInterp>
  </VRTRasterBand>
</VRTDataset>
'''
    # now write to file
    with open(vrt_path, 'w') as f:
        f.write(vrt_content)
    
    # call gdalwarp
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS='EPSG:4326',
        geoloc=True,
    )
    print(f"Running gdalwarp to produce {output_tif}...")
    gdal.Warp(destNameOrDestDS=output_tif, srcDSOrSrcDSTab=vrt_path, options=warp_options)

    print("Done")

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    OneKMPath = filedialog.askopenfilename(defaultextension='.hdf', title='Select 1km spectral data file')
    LatLongPath = filedialog.askopenfilename(defaultextension='.hdf', title='Select Lat/Long data file')
    bandIndex = 1
    try:
        startIndex = simpledialog.askinteger("Band Selection", "Enter Start Band Index, indexed at 1")
    except:
        print("Could not understand input. Default value of 1 will be used.")
    endIndex = startIndex
    try:
        endIndex = simpledialog.askinteger("Band Selection", "Enter End Band Index")
    except:
        print(f"Could not understand input. Default value of {startIndex} will be used")
    swath_indicator = 1
    try:
        swath_indicator = simpledialog.askinteger("Swath Selection", "Enter the indicator for swath desired:\n" \
        "1: 1km reflective\n" \
        "2: 500m aggrigate\n" \
        "3: 250m aggrigate\n" \
        "4: 1km emissive\n")
    except:
        print("Could not understand input. Default value of 1 will be used")
    output_folder = filedialog.askdirectory(title="Please specify a folder for the output file")
    output_tif = simpledialog.askstring(title="Output File", prompt="Enter Desired Output Filename")
    full_path = os.path.join(output_folder, output_tif)
    

    for i in range(startIndex, endIndex + 1):
        newPath = full_path + "band" + str(i) + ".tif"
        print(f"Creating tif for band {i} at {newPath}")
        create_vrt_and_warp(OneKMPath, LatLongPath, band_index=i, output_tif=newPath, swath_type_indicator=swath_indicator)
        print(f"Band {i} conversion complete")