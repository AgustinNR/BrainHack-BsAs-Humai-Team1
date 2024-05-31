# first line: 20
def _filter_and_extract(
    imgs,
    extraction_function,
    parameters,
    memory_level=0,
    memory=Memory(location=None),
    verbose=0,
    confounds=None,
    sample_mask=None,
    copy=True,
    dtype=None,
):
    """Extract representative time series using given function.

    Parameters
    ----------
    imgs : 3D/4D Niimg-like object
        Images to be masked. Can be 3-dimensional or 4-dimensional.

    extraction_function : function
        Function used to extract the time series from 4D data. This function
        should take images as argument and returns a tuple containing a 2D
        array with masked signals along with a auxiliary value used if
        returning a second value is needed.
        If any other parameter is needed, a functor or a partial
        function must be provided.

    For all other parameters refer to NiftiMasker documentation

    Returns
    -------
    signals : 2D numpy array
        Signals extracted using the extraction function. It is a scikit-learn
        friendly 2D array with shape n_samples x n_features.

    """
    # Since the calling class can be any *Nifti*Masker, we look for exact type
    if verbose > 0:
        class_name = enclosing_scope_name(stack_level=10)

    # If we have a string (filename), we won't need to copy, as
    # there will be no side effect
    imgs = stringify_path(imgs)
    if isinstance(imgs, str):
        copy = False

    if verbose > 0:
        print(
            f"[{class_name}] Loading data "
            f"from {_utils._repr_niimgs(imgs, shorten=False)}"
        )

    # Convert input to niimg to check shape.
    # This must be repeated after the shape check because check_niimg will
    # coerce 5D data to 4D, which we don't want.
    temp_imgs = _utils.check_niimg(imgs)

    # Raise warning if a 3D niimg is provided.
    if temp_imgs.ndim == 3:
        warnings.warn(
            "Starting in version 0.12, 3D images will be transformed to "
            "1D arrays. "
            "Until then, 3D images will be coerced to 2D arrays, with a "
            "singleton first dimension representing time.",
            DeprecationWarning,
        )

    imgs = _utils.check_niimg(
        imgs, atleast_4d=True, ensure_ndim=4, dtype=dtype
    )

    target_shape = parameters.get("target_shape")
    target_affine = parameters.get("target_affine")
    if target_shape is not None or target_affine is not None:
        if verbose > 0:
            print(f"[{class_name}] Resampling images")
        imgs = cache(
            image.resample_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
            ignore=["copy"],
        )(
            imgs,
            interpolation="continuous",
            target_shape=target_shape,
            target_affine=target_affine,
            copy=copy,
        )

    smoothing_fwhm = parameters.get("smoothing_fwhm")
    if smoothing_fwhm is not None:
        if verbose > 0:
            print(f"[{class_name}] Smoothing images")
        imgs = cache(
            image.smooth_img,
            memory,
            func_memory_level=2,
            memory_level=memory_level,
        )(imgs, parameters["smoothing_fwhm"])

    if verbose > 0:
        print(f"[{class_name}] Extracting region signals")
    region_signals, aux = cache(
        extraction_function,
        memory,
        func_memory_level=2,
        memory_level=memory_level,
    )(imgs)

    # Temporal
    # --------
    # Detrending (optional)
    # Filtering
    # Confounds removing (from csv file or numpy array)
    # Normalizing
    if verbose > 0:
        print(f"[{class_name}] Cleaning extracted signals")
    runs = parameters.get("runs", None)
    region_signals = cache(
        signal.clean,
        memory=memory,
        func_memory_level=2,
        memory_level=memory_level,
    )(
        region_signals,
        detrend=parameters["detrend"],
        standardize=parameters["standardize"],
        standardize_confounds=parameters["standardize_confounds"],
        t_r=parameters["t_r"],
        low_pass=parameters["low_pass"],
        high_pass=parameters["high_pass"],
        confounds=confounds,
        sample_mask=sample_mask,
        runs=runs,
        **parameters["clean_kwargs"],
    )

    return region_signals, aux
