# Importing Dependencies
# import pytest
# from imagestitcher import stitching_settings, rastering, scripts


# def test_walk_and_stitch():
#     """

#     """

#     settings = stitching_settings.StitchingSettings(ff_paths={},
#                                                     ff_params=None,
#                                                     setup_num=4,
#                                                     tile_dim=1024)

#     #TODO this looks terrible
#     settings.channels.update({'bf'})

#     multi_image_path = 'test/test_data/button_quant'

#     p = rastering.raster_params.RasterParams(overlap=.1,
#                                              size=settings.tile_dimensions,
#                                              acqui_ori=settings.acqui_ori,
#                                              rotation=0.42,
#                                              auto_ff=False)

#     scripts.walk_and_stitch(multi_image_path, settings, p, stitchtype='single')

#     assert 1==0
