# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Kaixu Yang
"""Util functions to pickle and load forecast results."""
import inspect
import os
import shutil
from collections import OrderedDict

import dill
from patsy.design_info import DesignInfo


def recursive_rm_dir(dir_name):
    """Recursively removes dirs and files in ``dir_name``.

    This functions removes everything in ``dir_name`` that it has permission
    to remove. This function is intended to remove the dumped directory.
    Do not use this function to remove other directories,
    unless you are sure to remove everything in the directory.

    Parameters
    ----------
    dir_name : `str`
        The directory name to be removed.

    Returns
    -------
    The functions removes the directory from local file system and does not return anything.
    """
    if os.path.isdir(dir_name):
        files = os.listdir(dir_name)
        if not files:
            os.rmdir(dir_name)
        else:
            [recursive_rm_dir(os.path.join(dir_name, file)) for file in files]
            os.rmdir(dir_name)
    else:
        os.remove(dir_name)


def dump_obj(
        obj,
        dir_name,
        obj_name="obj",
        dump_design_info=True,
        overwrite_exist_dir=False,
        top_level=True):
    """Uses DFS to recursively dump an object to pickle files.
    Originally intended for dumping the
    `~greykite.framework.pipeline.pipeline.ForecastResult` instance,
    but could potentially used for other objects.

    For each object, if it's picklable, a file with {object_name}.pkl will be
    generated, otherwise, depending on its type, a {object_name}.type file will
    be generated storing it's type, and a folder with {object_name} will be generated
    to store each of its elements/attributes.

    For example, if the folder to store results is forecast_result, the items in the
    folders could be:

        - timeseries.pkl: a picklable item.
        - model.type: model is not picklable, this file includes the class (Pipeline)
        - model: this folder includes the elements in model.
        - forecast.type: forecast is not picklable, this file includes the class (UnivariateForecast)
        - forecast: this folder includes the elements in forecast.
        - backtest.type: backtest is not picklable, this file includes the class (UnivariateForecast)
        - backtest: this folder includes the elements in backtest.
        - grid_search.type: grid_search is not picklable, this file includes the class (GridSearchCV)
        - grid_search: this folder includes the elements in grid_search.

    The items in each subfolder follows the same rule.

    The current supported recursion types are:

        - list/tuple: type name is "list" or "tuple", each element is attempted to
          be pickled independently if the entire list/tuple is not picklable.
          The order is preserved.
        - OrderedDict: type name is "ordered_dict", each key and value are attempted
          to be pickled independently if the entire dict is not picklable.
          The order is preserved.
        - dict: type name is "dict", each key and value are attempted to be pickled
          independently if the entire dict is not picklable.
          The order is not preserved.
        - class instance: type name is the class object, used to create new instance.
          Each attribute is attempted to be pickled independently if the entire
          instance is not picklable.

    Parameters
    ----------
    obj : `object`
        The object to be pickled.
    dir_name : `str`
        The directory to store the pickled results.
    obj_name : `str`, default "obj"
        The name for the pickled items. Applies to the top level object only
        when recursion is used.
    dump_design_info : `bool`, default True
        Whether to dump the design info in `ForecastResult`.
        The design info is specifically for Silverkite and can be accessed from

            - ForecastResult.model[-1].model_dict["x_design_info"]
            - ForecastResult.forecast.estimator.model_dict["x_design_info"]
            - ForecastResult.backtest.estimator.model_dict["x_design_info"]

        The design info is a class from `patsy` and contains a significant amount of
        instances that can not be pickled directly. Recursively pickling them takes
        longer to run. If speed is important and you don't need these information,
        you can turn it off.
    overwrite_exist_dir : `bool`, default False
        If True and the directory in ``dir_name`` already exists, the existing
        directory will be removed.
        If False and the directory in ``dir_name`` already exists, an exception
        will be raised.
    top_level : `bool`, default True
        Whether the implementation is an initial call
        (applies to the root object you want to pickle, not a recursive call).
        When you use this function to dump an object, this parameter should always be True.
        Only top level checks if the dir exists,
        because subsequent recursive calls may write files to the same directory,
        and the check for dir exists will not be implemented.
        Setting this parameter to False may cause problems.

    Returns
    -------
    The function writes files to local directory and does not return anything.
    """
    # Checks if to dump design info.
    if (not dump_design_info) and (isinstance(obj, DesignInfo) or (isinstance(obj, str) and obj == "x_design_info")):
        return

    # Checks if directory already exists.
    if top_level:
        dir_already_exist = os.path.exists(dir_name)
        if dir_already_exist:
            if not overwrite_exist_dir:
                raise FileExistsError("The directory already exists. "
                                      "Please either specify a new directory or "
                                      "set overwrite_exist_dir to True to overwrite it.")
            else:
                if os.path.isdir(dir_name):
                    # dir exists as a directory.
                    shutil.rmtree(dir_name)
                else:
                    # dir exists as a file.
                    os.remove(dir_name)

    # Creates the directory.
    # None top-level may write to the same directory,
    # so we allow existing directory in this case.
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

    # Start dumping recursively.
    try:
        # Attempts to directly dump the object.
        dill.dump(
            obj,
            open(os.path.join(dir_name, f"{obj_name}.pkl"), "wb"))
    except NotImplementedError:
        # Direct dumping fails.
        # Removed the failed file.
        try:
            os.remove(os.path.join(dir_name, f"{obj_name}.pkl"))
        except FileNotFoundError:
            pass
        # Attempts to do recursive dumping depending on the object type.
        if isinstance(obj, OrderedDict):
            # For OrderedDict (there are a lot in `pasty.design_info.DesignInfo`),
            # recursively dumps the keys and values, because keys can be class instances
            # and unpicklable, too.
            # The keys and values have index number appended to the front,
            # so the order is kept.
            dill.dump(
                "ordered_dict",
                open(os.path.join(dir_name, f"{obj_name}.type"), "wb"))  # type "ordered_dict"
            for i, (key, value) in enumerate(obj.items()):
                name = f"{i}_{str(key)}"
                dump_obj(
                    key,
                    os.path.join(dir_name, obj_name),
                    f"{name}__key__",
                    dump_design_info=dump_design_info,
                    top_level=False)
                dump_obj(
                    value,
                    os.path.join(dir_name, obj_name),
                    f"{name}__value__",
                    dump_design_info=dump_design_info,
                    top_level=False)
        elif isinstance(obj, dict):
            # For regular dictionary,
            # recursively dumps the keys and values, because keys can be class instances
            # and unpicklable, too.
            # The order is not important.
            dill.dump(
                "dict",
                open(os.path.join(dir_name, f"{obj_name}.type"), "wb"))  # type "dict"
            for key, value in obj.items():
                name = str(key)
                dump_obj(
                    key,
                    os.path.join(dir_name, obj_name),
                    f"{name}__key__",
                    dump_design_info=dump_design_info,
                    top_level=False)
                dump_obj(
                    value,
                    os.path.join(dir_name, obj_name),
                    f"{name}__value__",
                    dump_design_info=dump_design_info,
                    top_level=False)
        elif isinstance(obj, (list, tuple)):
            # For list and tuples,
            # recursively dumps the elements.
            # The names have index number appended to the front,
            # so the order is kept.
            dill.dump(
                type(obj).__name__,
                open(os.path.join(dir_name, f"{obj_name}.type"), "wb"))  # type "list"/"tuple"
            for i, value in enumerate(obj):
                dump_obj(
                    value,
                    os.path.join(dir_name, obj_name),
                    f"{i}_key",
                    dump_design_info=dump_design_info,
                    top_level=False)
        elif hasattr(obj, "__class__") and not isinstance(obj, type):
            # For class instance,
            # recursively dumps the attributes.
            dill.dump(
                obj.__class__,
                open(os.path.join(dir_name, f"{obj_name}.type"), "wb"))  # type is class itself
            for key, value in obj.__dict__.items():
                dump_obj(
                    value,
                    os.path.join(dir_name, obj_name),
                    key,
                    dump_design_info=dump_design_info,
                    top_level=False)
        else:
            # Other unrecognized unpicklable types, not common.
            print(f"I Don't recognize type {type(obj)}")


def load_obj(
        dir_name,
        obj=None,
        load_design_info=True):
    """Loads the pickled files which are pickled by
    `~greykite.framework.templates.pickle_utils.dump_obj`.
    Originally intended for loading the
    `~greykite.framework.pipeline.pipeline.ForecastResult` instance,
    but could potentially used for other objects.

    Parameters
    ----------
    dir_name : `str`
        The directory that stores the pickled files.
        Must be the top level dir when having nested pickling results.
    obj : `object`, default None
        The object type for the next-level files.
        Can be one of "list", "tuple", "dict", "ordered_dict" or a class.
    load_design_info : `bool`, default True
        Whether to load the design info in `ForecastResult`.
        The design info is specifically for Silverkite and can be accessed from

            - ForecastResult.model[-1].model_dict["x_design_info"]
            - ForecastResult.forecast.estimator.model_dict["x_design_info"]
            - ForecastResult.backtest.estimator.model_dict["x_design_info"]

        The design info is a class from `patsy` and contains a significant amount of
        instances that can not be pickled directly. Recursively loading them takes
        longer to run. If speed is important and you don't need these information,
        you can turn it off.

    Returns
    -------
    result : `object`
        The loaded object from the pickled files.
    """
    # Checks if to load design info.
    if (not load_design_info) and (isinstance(obj, type) and obj == DesignInfo):
        return None

    # Gets file names in the level.
    files = os.listdir(dir_name)
    if not files:
        raise ValueError("dir is empty!")

    # Gets the type files if any.
    # Stores in a dictionary with key being the name and value being the loaded value.
    obj_types = {file.split(".")[0]: dill.load(open(os.path.join(dir_name, file), "rb"))
                 for file in files if ".type" in file}

    # Gets directories and pickled files.
    # Every type must have a directory with the same name.
    directories = [file for file in files if os.path.isdir(os.path.join(dir_name, file))]
    if not all([directory in obj_types for directory in directories]):
        raise ValueError("type and directories do not match.")
    pickles = [file for file in files if ".pkl" in file]

    # Starts loading objects
    if obj is None:
        # obj is None indicates this is the top level directory.
        # This directory can either have 1 .pkl file, or 1 .type file associated with the directory of same name.
        if not obj_types:
            # The only 1 .pkl file case.
            if len(files) > 1:
                raise ValueError("Multiple elements found in top level.")
            return dill.load(open(os.path.join(dir_name, files[0]), "rb"))
        else:
            # The .type + dir case.
            if len(obj_types) > 1:
                raise ValueError("Multiple elements found in top level")
            obj_name = list(obj_types.keys())[0]
            obj_type = obj_types[obj_name]
            return load_obj(
                os.path.join(dir_name, obj_name),
                obj_type,
                load_design_info=load_design_info)
    else:
        # If obj is not None, does recursive loading depending on the obj type.
        if obj in ("list", "tuple"):
            # Object is list or tuple.
            # Fetches each element according to the number index to preserve orders.
            result = []
            # Order index is a number appended to the front.
            elements = sorted(
                pickles + directories,
                key=lambda x: int(x.split("_")[0]))
            # Recursively loads elements.
            for element in elements:
                if ".pkl" in element:
                    result.append(
                        dill.load(open(os.path.join(dir_name, element), "rb")))
                else:
                    result.append(
                        load_obj(
                            os.path.join(dir_name, element),
                            obj_types[element],
                            load_design_info=load_design_info))
            if obj == "tuple":
                result = tuple(result)
            return result
        elif obj == "dict":
            # Object is a dictionary.
            # Fetches keys and values recursively.
            result = {}
            elements = pickles + directories
            keys = [element for element in elements if "__key__" in element]
            values = [element for element in elements if "__value__" in element]
            # Iterates through keys and finds the corresponding values.
            for element in keys:
                if ".pkl" in element:
                    key = dill.load(
                        open(os.path.join(dir_name, element), "rb"))
                else:
                    key = load_obj(
                        os.path.join(dir_name, element),
                        obj_types[element],
                        load_design_info=load_design_info)
                # Value name could be either with .pkl or a directory.
                value_name = element.replace("__key__", "__value__")
                if ".pkl" in value_name:
                    value_name_alt = value_name.replace(".pkl", "")
                else:
                    value_name_alt = value_name + ".pkl"
                # Checks if value name is in the dir.
                if (value_name not in values) and (value_name_alt not in values):
                    raise FileNotFoundError(f"Value not found for key {key}.")
                value_name = value_name if value_name in values else value_name_alt
                # Gets the value.
                if ".pkl" in value_name:
                    value = dill.load(
                        open(os.path.join(dir_name, value_name), "rb"))
                else:
                    value = load_obj(
                        os.path.join(dir_name, value_name),
                        obj_types[value_name],
                        load_design_info=load_design_info)
                # Sets the key, value pair.
                result[key] = value
            return result
        elif obj == "ordered_dict":
            # Object is OrderedDict.
            # Fetches keys and values according to the number index to preserve orders.
            result = OrderedDict()
            # Order index is a number appended to the front.
            elements = sorted(pickles + directories, key=lambda x: int(x.split("_")[0]))
            keys = [element for element in elements if "__key__" in element]
            values = [element for element in elements if "__value__" in element]
            # Iterates through keys and finds the corresponding values.
            for element in keys:
                if ".pkl" in element:
                    key = dill.load(
                        open(os.path.join(dir_name, element), "rb"))
                else:
                    key = load_obj(
                        os.path.join(dir_name, element),
                        obj_types[element],
                        load_design_info=load_design_info)
                value_name = element.replace("__key__", "__value__")
                # Value name could be either with .pkl or a directory.
                if ".pkl" in value_name:
                    value_name_alt = value_name.replace(".pkl", "")
                else:
                    value_name_alt = value_name + ".pkl"
                # Checks if value name is in the dir.
                if (value_name not in values) and (value_name_alt not in values):
                    raise FileNotFoundError(f"Value not found for key {key}.")
                value_name = value_name if value_name in values else value_name_alt
                # Gets the value.
                if ".pkl" in value_name:
                    value = dill.load(
                        open(os.path.join(dir_name, value_name), "rb"))
                else:
                    value = load_obj(
                        os.path.join(dir_name, value_name),
                        obj_types[value_name],
                        load_design_info=load_design_info)
                # Sets the key, value pair.
                result[key] = value
            return result
        elif inspect.isclass(obj):
            # Object is a class instance.
            # Creates the class instance and sets the attributes.
            # Some class has required args during initialization,
            # these args are pulled from attributes.
            init_params = list(inspect.signature(obj.__init__).parameters)  # init args
            elements = pickles + directories
            # Gets the attribute names and their values in a dictionary.
            values = {}
            for element in elements:
                if ".pkl" in element:
                    values[element.split(".")[0]] = dill.load(
                        open(os.path.join(dir_name, element), "rb"))
                else:
                    values[element] = load_obj(
                        os.path.join(dir_name, element),
                        obj_types[element],
                        load_design_info=load_design_info)
            # Gets the init args from values.
            init_dict = {key: value for key, value in values.items()
                         if key in init_params}
            # Some attributes has a "_" at the beginning.
            init_dict.update({key[1:]: value for key, value in values.items()
                              if (key[1:] in init_params and key[0] == "_")})
            # ``design_info`` does not have column_names attribute,
            # which is required during init.
            # The column_names param is pulled from the column_name_indexes attribute.
            # This can be omitted once we allow dumping @property attributes.
            if "column_names" in init_params:
                init_dict["column_names"] = values["column_name_indexes"].keys()
            # Creates the instance.
            result = obj(**init_dict)
            # Sets the attributes.
            for key, value in values.items():
                setattr(result, key, value)
            return result
        else:
            # Raises an error if the object is not recognized.
            # This typically does not happen when the source file is dumped
            # with the `dump_obj` function.
            raise ValueError(f"Object {obj} is not recognized.")
