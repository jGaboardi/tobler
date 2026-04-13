"""Model-based methods for areal interpolation."""
import geopandas as gpd
import numpy as np  # noqa: F401 -- needed for numpy's logarithm in the formula string
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, NegativeBinomial, Poisson

from ..util.util import _check_presence_of_crs, _rename_type
from .raster_tools import extract_raster_features

__all__ = ["glm"]


def glm(
    source_df=None,
    target_df=None,
    raster="nlcd_2011",
    pixel_values=None,
    variable=None,
    formula=None,
    likelihood="poisson",
    force_crs_match=True,
    return_model=False,
    nodata=255,
    n_jobs=-1
):
    """Train a generalized linear model to predict polygon attributes
    based on the collection of pixel values they contain.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame, required
        geodataframe containing source original data to
        be represented by another geometry
    target_df : geopandas.GeoDataFrame, required
        geodataframe containing target boundaries that
        will be used to represent the source data
    raster : str, required (default="nlcd_2011")
        path to raster file that will be used to input data to the regression model.
        i.e. a coefficients refer to the relationship between pixel
        counts and population counts. Defaults to 2011 NLCD
    pixel_values : list, required (default =[21, 22, 23, 24, 41, 42, 52])
        list of integers that represent different types of raster cells. If no formula
        is given, the model will be fit from a linear combination of the logged count
        of each cell type listed here. Defaults to [21, 22, 23, 24, 41, 42, 52] which
        are informative land type cells from the NLCD
    variable : str, required
        name of the variable (column) to be modeled from the `source_df`
    formula : str, optional
        patsy-style model formula that specifies the model. Raster codes should be
        prefixed with "Type_", e.g.
        `"n_total_pop ~ -1 + Type_21 + Type_22`
    likelihood : str, {'poisson', 'gaussian', 'neg_binomial'} (default = "poisson")
        the likelihood function used in the model
    force_crs_match : bool
        whether to coerce geodataframe and raster to the same CRS
    return model : bool
        whether to return the fitted model in addition to the interpolated geodataframe.
        If true, this will return (geodataframe, model)
    nodata : int
        value in raster that indicates null or missing values. Default is 255
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel to
        generate the area allocation. If -1, this is set to the number of CPUs
        available.

    Returns
    --------
    interpolated : geopandas.GeoDataFrame
        a new geopandas dataframe with boundaries from `target_df` and modeled
        attribute data from the `source_df`. If `return_model` is true, the
        function will also return the fitted regression model for further diagnostics
        The new geopandas will present a column named "GLM_PRED_" + variable
    """
    source_df = source_df.copy()
    target_df = target_df.copy()
    _check_presence_of_crs(source_df)
    liks = {"poisson": Poisson, "gaussian": Gaussian, "neg_binomial": NegativeBinomial}

    if likelihood not in liks:
        raise ValueError(f"likelihood must one of {liks.keys()}")

    if not pixel_values:
        pixel_values = [21, 22, 23, 24, 41, 42, 52]
    pixel_values_str = ["Type_" + str(i) for i in pixel_values]

    if not formula:
        formula = (
            variable
            + "~ -1 +"
            + " + ".join([code for code in pixel_values_str])
        )

    #  create a vector mask from the raster data
    # Pass collapse_values=False to obtain each pixel type
    raster_mask = extract_raster_features(
        source_df, raster, pixel_values, nodata, n_jobs, collapse_values=False
    )
    
    raster_mask = raster_mask.copy()
    
    source_df["source_id"] = source_df.index
    target_df["target_id"] = target_df.index
    raster_mask["mask_id"] = raster_mask.index
    
    # Intersect the two layers (source + raster)
    source_intersections = gpd.overlay(
        raster_mask[["mask_id", "value", "geometry"]],
        source_df[["source_id", "geometry"]],
        how="intersection"
    ).dissolve("mask_id").reset_index()

    # Count how many intersected pieces of each category fall in each source polygon
    source_counts = (
        source_intersections.groupby(["source_id", "value"])
                     .size()
                     .unstack(fill_value=0)
                     .reset_index()
    )

    source_result = source_df.merge(source_counts, on="source_id", how="left").fillna(0)
    source_result = _rename_type(source_result)
    results_regression = smf.glm(formula, data=source_result, family=liks[likelihood]()).fit()
    
    # Extract the log from predict to get the weights sums
    # "raw" is because this prediction is only generated through the regression (that is, without enforcing pycnophylactic property)
    source_result['pred_variable_on_source_id_raw'] = np.log(results_regression.predict(source_result.fillna(0)))
    
    # Append source predictions and original variable for the pycnophylactic property step
    source_intersections_2 = source_intersections.\
        merge(source_result[['source_id', variable, 'pred_variable_on_source_id_raw']], 
              on="source_id", 
              how="left").\
        fillna(0)
        
    # Match each pixel to its corresponding weight
    coefs = results_regression.params
    type_coefs = coefs[coefs.index.str.match(r"^Type_\d+$")]
    mapping = { # build mapping: 21 -> coef(Type_21), 22 -> coef(Type_22), ...
        int(name.split("_", 1)[1]): coef
        for name, coef in type_coefs.items()
    }

    # add coefficient as a new column
    source_intersections_2["type_coef"] = source_intersections_2["value"].map(mapping)
    
    # Ensure pycnophylactic property
    source_intersections_2['pred_variable_on_pixel'] = source_intersections_2['type_coef'] * (source_intersections_2[variable] / source_intersections_2['pred_variable_on_source_id_raw'])
    
    # Intersect the two layers (target + raster)   
    target_intersections = gpd.overlay(
        raster_mask[["mask_id", "geometry"]],
        target_df[["target_id", "geometry"]],
        how="intersection"
    ).dissolve("mask_id").reset_index()
    
    # Recover pixel estimated population for each mask id within every target id
    target_intersections_2 = target_intersections.\
        merge(source_intersections_2[['mask_id', 'pred_variable_on_pixel']], 
              on="mask_id", 
              how="inner")
        
    # Sum the weights for each target polygon to get the estimated population with pycnophylactic property
    sum_by_target = (
        target_intersections_2.groupby("target_id", as_index=False)["pred_variable_on_pixel"]
          .sum()
    )
    
    # Append these estimates into the original target geopandas
    interpolated = target_df.\
        merge(sum_by_target, 
              on="target_id", 
              how="inner").\
        rename(columns={"pred_variable_on_pixel": "GLM_PRED_" + variable})

    if return_model:
        return interpolated, results_regression

    return interpolated
