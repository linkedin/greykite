Model store and load
====================

It's common that people would like to store a trained model to use in the future.
Usually the class instances are not easy to dump directly, due to the
complexity of the instance structure.
However, Greykite has utility functions to store and load trained models as well as other
objects in a recursive way.
Simply speaking, Greykite tries to dump a class instance directly.
If it fails, Greykite will store the class type and every single attribute it has, recursively.
When the stored class instance is being loaded,
these information are used to reconstruct the class instances.

Storing and loading the Forecaster class
----------------------------------------
The `~greykite.framework.templates.forecaster.Forecaster` class is the API for
all Greykite algorithms. A typical training flow is as below.

.. code-block:: python

    forecaster = Forecaster()
    forecaster.run_forecast_config(
        df=df,
        config=config)

After training, all training information will be accessible via ``forecaster.forecast_result``.
The ``Forecaster`` class has natively built-in dump and load functions,
`~greykite.framework.templates.forecaster.Forecaster.dump_forecast_result` and
`~greykite.framework.templates.forecaster.Forecaster.load_forecast_result`.

Dump the results
^^^^^^^^^^^^^^^^

To store the trained model, we can directly use

.. code-block:: python

    forecaster.dump_forecast_result(
        destination_dir,
        object_name="object",
        dump_design_info=True,
        overwrite_exist_dir=False)

The ``destination_dir`` is a string of the saving location.
The ``object_name`` is the name of the stored folder.
The ``dump_design_info`` is specifically for the Silverkite model family.
Setting it to ``False`` to avoid dumping the design info of the design matrix.
The design info is not useful in general (to reproduce results, make predictions),
and not dumping it will save a lot of time.
Specify ``overwrite_exist_dir=True`` to force overwriting when a file to be
written already exists.

This ``dump_forecast_result`` function will save the entire ``forecaster.forecast_result`` object
as a folder at the specified directory. Typically a saved model takes 50+ MB, so please make
sure you have enough storage space.

Load the results
^^^^^^^^^^^^^^^^

Assume you have the dumped results in a specified directory named ``source_dir``,
you can reconstruct the ``Forecaster`` class by

.. code-block:: python

    forecaster = Forecaster()
    forecaster.load_forecast_result(
        source_dir,
        load_design_info=True)

The ``source_dir`` is a string of the loading location.
The ``load_design_info`` is specifically for the Silverkite model family.
Setting it to ``False`` to avoid loading the design info of the design matrix.
The design info is not useful in general (to reproduce results, make predictions),
and not dumping it will save a lot of time.
However, if the design info was not dumped, this parameter will have no effect.

This ``load_forecast_result`` function will reconstruct your trained ``Forecaster`` class.

Storing and loading a general object
------------------------------------

If you want to store or load a general object, you can try the
`~greykite.framework.templates.pickle_utils.dump_obj` and
`~greykite.framework.templates.pickle_utils.load_obj` functions.
These function are the utility function used by the ``Forecaster`` class
to dump and load the ``forecast_result`` objects,
however, they can be potentially used for other objects as well.

Assume we have a class instance named ``model``, then we can dump it to ``dir_name`` with

.. code-block:: python

    dump_obj(
        model,
        dir_name,
        obj_name="model",
        dump_design_info=True,
        overwrite_exist_dir=False)

Note that if the model has nothing to do with the Silverkite model family,
the ``dump_design_info`` parameter is ignored.

Similarly, we can load the object from ``source_dir`` with

.. code-block:: python

    model = load_obj(
        source_dir,
        load_design_info=True)
