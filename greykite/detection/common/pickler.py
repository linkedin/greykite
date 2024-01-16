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
# original author: Sayan Patra, Kaixu Yang

"""Util functions to serialize and deserialize anomaly detector."""
import codecs
import inspect
from collections import OrderedDict

import dill


# Constants used for serialization of the model.
# Be careful not to have these strings in the object to be serialized.
KEY_SUFFIX = "__key__"
"""Suffix used to denote the key of a dictionary during serialization."""
VALUE_SUFFIX = "__value__"
"""Suffix used to denote the value of a dictionary during serialization."""
OBJECT_TYPE_ROOT = "ROOT"
"""The string to denote the root of the serialized object tree."""
PICKLE_KEY_EXTENSION = ".pkl"
"""The extension used to denote a object that can be directly serialized by dill."""
DIRECTORY_KEY_EXTENSION = ".dir"
"""The extension used to denote a directory that can not be directly serialized by dill."""
TYPE_KEY_EXTENSION = ".type"
"""The extension used to denote the type of an object that can not be directly serialized by dill."""


class GreykitePickler:
    """Extends the functionality of dill to serialize arbitrary objects.

    Originally intended to serialize the anomaly detector class
    `~greykite.detection.anomaly_detector.AnomalyDetector`, but
    can be potentially used to serialize other objects.

    Outputs a dictionary that can be serialized by `json` or `yaml` libraries.
    The dictionary has a tree structure, where each node is either a
    serialized object, or a pair of type and directory of serialized objects.
    The base of the directory is called "ROOT".

    Usage
    -----
        # To serialize
        pickler = GreykitePickler()
        pickled = pickler.dumps(obj)
        pickled_json = json.dumps(pickled)

        # To deserialize
        pickled = json.loads(pickled_json)
        obj = pickler.loads(pickled)

    Attributes
    ----------
    obj: any
        The object to be serialized.
    """
    def __init__(self):
        """Initializes an instance of the GreykitePickler class."""
        self.obj = None

    def dumps(self, obj, obj_name=OBJECT_TYPE_ROOT):
        """Uses Depth First Search (DFS) to recursively serialize the input `obj`.

        For each object, the following happens:
        1. If the `obj` is serializable by `dill`, a dictionary key with {`obj_name`}.pkl will be generated.

            ```python
            {
                "{obj_name}.pkl": "serialized object",
            }
            ```

        2. Else a dictionary with two keys are generated.
            - {`obj_name`}.type stores the object type.
            - {`obj_name`}.dir stores the elements/attributes of the object.
            To build the {`obj_name`}.dir dictionary, `dumps` method is called recursively.\

            ```python
            {
                "{obj_name}.type": "type of the object",
                "{obj_name}.dir": {
                    "key1": "serialized value1",
                    "key2": "serialized value2",
                    ...
                }
            }
            ```

        The current supported recursion types are:

        - list/tuple: type name is "list" or "tuple", each element is attempted to
          be pickled independently if the entire list/tuple is not serializable.
          The order is preserved.
        - OrderedDict: type name is "ordered_dict", each key and value are attempted
          to be pickled independently if the entire dict is not serializable.
          The order is preserved.
        - dict: type name is "dict", each key and value are attempted to be pickled
          independently if the entire dict is not serializable.
          The order is not preserved.
        - class instance: type name is the class object, used to create new instance.
          Each attribute is attempted to be pickled independently if the entire
          instance is not serializable.

        Parameters
        ----------
        obj: any
            The object to be serialized.
        obj_name: `str`, default "ROOT"
            The name of the object to be serialized. Default is "ROOT".

        Returns
        -------
        serialized: `dict`
            The serialized object.

        Raises
        ------
        NotImplementedError: If the object cannot be serialized.
        """
        self.obj = obj
        try:
            serialized = self.dumps_to_str(obj)
            return {f"{obj_name}{PICKLE_KEY_EXTENSION}": serialized}
        except NotImplementedError:
            if isinstance(obj, OrderedDict):
                return self._serialize_ordered_dict(obj, obj_name)
            if isinstance(obj, dict):
                return self._serialize_dict(obj, obj_name)
            if isinstance(obj, (list, tuple)):
                return self._serialize_list_tuple(obj, obj_name)
            if hasattr(obj, "__class__") and not isinstance(obj, type):
                return self._serialize_class(obj, obj_name)
            else:
                raise NotImplementedError(f"Cannot pickle object of type {type(obj)}.")

    def loads(self, serialized_dict, obj_type=OBJECT_TYPE_ROOT):
        """Deserializes the output of the `dumps` method.

        Parameters
        ----------
        serialized_dict: `dict`
            The output of the `dumps` method.
        obj_type: `str`, default "ROOT"
            The type of the object to be deserialized.

        Returns
        -------
        obj: any
            The deserialized object.

        Raises
        ------
        NotImplementedError: If the object cannot be deserialized.
        """
        if obj_type == OBJECT_TYPE_ROOT:
            return self._deserialize_root(serialized_dict, obj_type)
        if obj_type == OrderedDict.__name__:
            return self._deserialize_ordered_dict(serialized_dict, obj_type)
        if obj_type == dict.__name__:
            return self._deserialize_dict(serialized_dict, obj_type)
        if obj_type in ("list", "tuple"):
            return self._deserialize_list_tuple(serialized_dict, obj_type)
        if inspect.isclass(obj_type):
            return self._deserialize_class(serialized_dict, obj_type)
        else:
            raise NotImplementedError(f"Cannot unpickle object of type {obj_type}.")

    def _serialize_ordered_dict(self, obj, obj_name):
        """Pickles an ordered dictionary when it can not be directly pickled by `dill`.

        Generates a dictionary with two keys.
            - {`obj_name`}.type stores the object type.
            - {`obj_name`}.dir stores the elements/attributes of the object.
            To build the {`obj_name`}.dir dictionary, `dumps` method is called recursively.
            The keys are serialized with the suffix "__key__" and prefix "{key_order}" to preserve the order.
            Similarly, for the values, the suffix "__value__" is used.

            ```python
            {
                "{obj_name}.type": "serializable object type",
                "{obj_name}.dir": {
                    "0_{key1}__key__": "serialized key1",
                    "0_{key1}__value__": "serialized value corresponding to key1",
                    "1_{key2}__key__": "serialized key2",
                    "1_{key2}__value__": "serialized value corresponding to key2",
                    ...
                }
            }
            ```

        Parameters
        ----------
        obj: `OrderedDict`
            The ordered dictionary to be serialized.
        obj_name: `str`
            The name of the object to be serialized.

        Returns
        -------
        serialized: `dict`
            The serialized ordered dictionary.
        """
        # Dumps the type in .type dictionary
        serialized = {f"{obj_name}{TYPE_KEY_EXTENSION}": self.dumps_to_str(type(obj).__name__)}

        # Dumps the keys and values in .dir dictionary
        result = {}
        for i, (key, value) in enumerate(obj.items()):
            name = f"{i}_{str(key)}"
            result.update(self.dumps(obj=key, obj_name=f"{name}{KEY_SUFFIX}"))
            result.update(self.dumps(obj=value, obj_name=f"{name}{VALUE_SUFFIX}"))
        serialized[f"{obj_name}{DIRECTORY_KEY_EXTENSION}"] = result

        return serialized

    def _serialize_dict(self, obj, obj_name):
        """Pickles a dictionary when it can not be directly pickled by `dill`.

        Generates a dictionary with two keys.
            - {`obj_name`}.type stores the object type.
            - {`obj_name`}.dir stores the elements/attributes of the object.
            To build the {`obj_name`}.dir dictionary, `dumps` method is called recursively.
            The keys are serialized with the suffix "__key__" and the values are
            serialized with the suffix "__value__".
            This is done because the keys of a dictionary can be complex classes that can
            not be directly serialized by `dill`.

            ```python
            {
                "{obj_name}.type": "serializable object type",
                "{obj_name}.dir": {
                    "{key1}__key__": "serialized key1",
                    "{key1}__value__": "serialized value corresponding to key1",
                    "{key2}__key__": "serialized key2",
                    "{key2}__value__": "serialized value corresponding to key2",
                    ...
                }
            }
            ```

        Parameters
        ----------
        obj: `dict`
            The dictionary to be serialized.
        obj_name: `str`
            The name of the object to be serialized.

        Returns
        -------
        serialized: `dict`
            The serialized dictionary.
        """
        # Dumps the type in .type dictionary
        serialized = {f"{obj_name}{TYPE_KEY_EXTENSION}": self.dumps_to_str(type(obj).__name__)}

        # Dumps the keys and values in .dir dictionary
        result = {}
        for key, value in obj.items():
            name = str(key)
            result.update(self.dumps(obj=key, obj_name=f"{name}{KEY_SUFFIX}"))
            result.update(self.dumps(obj=value, obj_name=f"{name}{VALUE_SUFFIX}"))
        serialized[f"{obj_name}{DIRECTORY_KEY_EXTENSION}"] = result

        return serialized

    def _serialize_list_tuple(self, obj, obj_name):
        """Serializes a list or a tuple, preserving its order, when it can not be directly pickled by `dill`.

        Generates a dictionary with two keys.
            - {`obj_name`}.type stores the object type.
            - {`obj_name`}.dir stores the elements/attributes of the object.
            To build the {`obj_name`}.dir dictionary, `dumps` method is called recursively.

            ```python
            {
                "{obj_name}.type": "serializable object type",
                "{obj_name}.dir": {
                    "0__key__": "serialized value1",
                    "1__key__": "serialized value2",
                    ...
                }
            }
            ```

        Parameters
        ----------
        obj: `list` or `tuple`
            The list or tuple to be serialized.
        obj_name: `str`
            The name of the object to be serialized.

        Returns
        -------
        serialized: `dict`
            The serialized list or tuple.
        """
        # Dumps the type in .type dictionary
        serialized = {f"{obj_name}{TYPE_KEY_EXTENSION}": self.dumps_to_str(type(obj).__name__)}

        # Dumps the keys and values in .dir dictionary
        result = {}
        for i, value in enumerate(obj):
            result.update(self.dumps(obj=value, obj_name=f"{i}{KEY_SUFFIX}"))
        serialized[f"{obj_name}{DIRECTORY_KEY_EXTENSION}"] = result

        return serialized

    def _serialize_class(self, obj, obj_name):
        """Pickles a class when it can not be directly pickled by `dill`.

        Generates a dictionary with two keys.
            - {`obj_name`}.type stores the object type.
            - {`obj_name`}.dir stores the elements/attributes of the object.
            To build the {`obj_name`}.dir dictionary, `dumps` method is called recursively.

            ```python
            {
                "{obj_name}.type": "serializable object type",
                "{obj_name}.dir": {
                    "{key1}": "serialized value corresponding to key1",
                    "{key2}": "serialized value corresponding to key2",
                    ...
                }
            }
            ```
            Unlike `dict` and `OrderedDict`, the keys of the class attributes does not need to be
            serialized, as these are simple strings.

        Parameters
        ----------
        obj: `class`
            The class to be serialized.
        obj_name: `str`
            The name of the object to be serialized.

        Returns
        -------
        serialized: `dict`
            The serialized calss.
        """
        # Dumps the type in .type key
        serialized = dict()
        serialized[f"{obj_name}{TYPE_KEY_EXTENSION}"] = self.dumps_to_str(obj.__class__)

        # Initiates the .dir dictionary
        serialized[f"{obj_name}{DIRECTORY_KEY_EXTENSION}"] = {}
        # Dumps the class attributes in .dir key
        for key, value in obj.__dict__.items():
            serialized[f"{obj_name}{DIRECTORY_KEY_EXTENSION}"].update(self.dumps(obj=value, obj_name=key))

        return serialized

    def _deserialize_root(self, serialized_dict, obj_type):
        """Deserializes the root object.
        This is the very top level of the nested `serialized_dict`.
        Thus, it either contains a single .pkl file or a single .type + .dir pair.

        Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.
        obj_type: `str`
            The type of the object to be deserialized. Must be "ROOT".

        Returns
        -------
        obj: any
            The deserialized object.
        """
        if obj_type != OBJECT_TYPE_ROOT:
            raise ValueError(f"The obj_type must be {OBJECT_TYPE_ROOT}.")

        pickles, directories, obj_types = self._get_keys_from_serialized_dict(serialized_dict)
        if len(pickles) > 1 or len(directories) > 1:
            raise ValueError("Multiple elements found in the top level.")
        if f"{OBJECT_TYPE_ROOT}{PICKLE_KEY_EXTENSION}" in pickles:
            # The only 1 .pkl file case.
            return self.loads_from_str(serialized_dict[f"{OBJECT_TYPE_ROOT}{PICKLE_KEY_EXTENSION}"])
        else:
            # The .type + .dir case.
            return self.loads(
                serialized_dict[f"{OBJECT_TYPE_ROOT}{DIRECTORY_KEY_EXTENSION}"],
                obj_types[OBJECT_TYPE_ROOT])

    def _deserialize_ordered_dict(self, serialized_dict, obj_type):
        """Deserializes an ordered dictionary.

        Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.
        obj_type: `str`
            The type of the object to be deserialized. Must be "OrderedDict".

        Returns
        -------
        obj: `OrderedDict`
            The deserialized ordered dictionary.
        """
        if obj_type != OrderedDict.__name__:
            raise ValueError("The obj_type must be OrderedDict.")

        pickles, directories, obj_types = self._get_keys_from_serialized_dict(serialized_dict)
        # Object is a OrderedDict.
        # Fetch keys and values according to the number index to preserve orders.
        result = OrderedDict()
        # Order index is a number appended to the front.
        elements = sorted(pickles + directories, key=lambda x: int(x.split("_")[0]))
        keys = [element for element in elements if KEY_SUFFIX in element]
        for element in keys:
            if PICKLE_KEY_EXTENSION in element:
                key = self.loads_from_str(serialized_dict[element])
            else:
                key = self.loads(
                    serialized_dict[element],
                    obj_types[element.split(".")[0]])

            # Searches for the value corresponding to the key.
            element = element.replace(KEY_SUFFIX, VALUE_SUFFIX).split(".")[0]
            # Value name could be either with .pkl or a directory.
            if f"{element}{PICKLE_KEY_EXTENSION}" in pickles:
                element = f"{element}{PICKLE_KEY_EXTENSION}"
                value = self.loads_from_str(serialized_dict[element])
            elif f"{element}{DIRECTORY_KEY_EXTENSION}" in directories:
                element = f"{element}{DIRECTORY_KEY_EXTENSION}"
                value = self.loads(
                    serialized_dict[element],
                    obj_types[element.split(".")[0]])
            else:
                raise ValueError(f"Value not found for key {key}.")
            # Sets the key, value pair.
            result[key] = value

        return result

    def _deserialize_dict(self, serialized_dict, obj_type):
        """Deserializes a dictionary.

        Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.
        obj_type: `str`
            The type of the object to be deserialized. Must be "dict".

        Returns
        -------
        obj: `dict`
            The deserialized dictionary.
        """
        if obj_type != dict.__name__:
            raise ValueError("The obj_type must be dict.")

        pickles, directories, obj_types = self._get_keys_from_serialized_dict(serialized_dict)
        result = {}
        elements = pickles + directories
        keys = [element for element in elements if KEY_SUFFIX in element]
        # Iterates through keys and finds the corresponding values.
        for element in keys:
            if PICKLE_KEY_EXTENSION in element:
                key = self.loads_from_str(serialized_dict[element])
            else:
                key = self.loads(
                    serialized_dict[element],
                    obj_types[element.split(".")[0]])

            # Searches for the value corresponding to the key.
            element = element.replace(KEY_SUFFIX, VALUE_SUFFIX).split(".")[0]
            # Value name could be either with .pkl or a directory.
            if f"{element}{PICKLE_KEY_EXTENSION}" in pickles:
                element = f"{element}{PICKLE_KEY_EXTENSION}"
                value = self.loads_from_str(serialized_dict[element])
            elif f"{element}{DIRECTORY_KEY_EXTENSION}" in directories:
                element = f"{element}{DIRECTORY_KEY_EXTENSION}"
                value = self.loads(
                    serialized_dict[element],
                    obj_types[element.split(".")[0]])
            else:
                raise ValueError(f"Value not found for key {key}.")
            # Sets the key, value pair.
            result[key] = value

        return result

    def _deserialize_list_tuple(self, serialized_dict, obj_type):
        """Deserializes a list or a tuple.

        Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.
        obj_type: `str`
            The type of the object to be deserialized. Must be "list" or "tuple".

        Returns
        -------
        obj: `list` or `tuple
            The deserialized list or tuple.
        """
        pickles, directories, obj_types = self._get_keys_from_serialized_dict(serialized_dict)
        result = []
        # Order index is a number appended to the front.
        elements = sorted(pickles + directories, key=lambda x: int(x.split("_")[0]))
        # Recursively loads elements.
        for element in elements:
            if PICKLE_KEY_EXTENSION in element:
                value = self.loads_from_str(serialized_dict[element])
            else:
                value = self.loads(
                    serialized_dict[element],
                    obj_types[element.split(".")[0]])
            result.append(value)

        if obj_type == "tuple":
            result = tuple(result)

        return result

    def _deserialize_class(self, serialized_dict, obj_type):
        """Deserializes a class.

         Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.
        obj_type: `str`
            The type of the object to be deserialized. Must be instance of a class.

        Returns
        -------
        obj: `class`
            The deserialized class.
        """
        pickles, directories, obj_types = self._get_keys_from_serialized_dict(serialized_dict)
        # Object is a class instance.
        # Creates the class instance and sets the attributes.
        # Some class has required args during initialization,
        # these args are pulled from attributes.
        init_params = list(inspect.signature(obj_type.__init__).parameters)  # init args
        elements = pickles + directories
        # Gets the attribute names and their values in a dictionary.
        values = {}
        for element in elements:
            if PICKLE_KEY_EXTENSION in element:
                values[element.split(".")[0]] = self.loads_from_str(serialized_dict[element])
            else:
                values[element.split(".")[0]] = self.loads(
                    serialized_dict=serialized_dict[element],
                    obj_type=obj_types[element.split(".")[0]],
                )
        # Gets the init args from values.
        init_dict = {key: value for key, value in values.items() if key in init_params}
        # Some attributes have a "_" at the beginning.
        init_dict.update({key[1:]: value for key, value in values.items()
                          if (key[1:] in init_params and key[0] == "_")})
        # ``design_info`` does not have column_names attribute,
        # which is required during init.
        # The column_names param is pulled from the column_name_indexes attribute.
        # This can be omitted once we allow dumping @property attributes.
        if "column_names" in init_params:
            init_dict["column_names"] = values["column_name_indexes"].keys()
        # Creates the instance.
        result = obj_type(**init_dict)
        # Sets the attributes.
        for key, value in values.items():
            setattr(result, key, value)

        return result

    @staticmethod
    def dumps_to_str(obj):
        """Returns a serialized string representation of the `obj`.
        The `obj` must be serializable by `dill`.

        Serialized output `dill` is a bytes object, which can not be stored in a json file.
        This method encodes the bytes object to a base64 string, which can be stored in a json file.

        Parameters
        ----------
        obj: any
            The object to be serialized. Must be serializable by `dill`.

        Returns
        -------
        serialized_string: `str`
            A serialized string representation of the object.
        """
        return codecs.encode(dill.dumps(obj), "base64").decode()

    @staticmethod
    def loads_from_str(serialized_string):
        """Returns a deserialized object from a `serialized_string`, usually the
        output of `dumps_to_str`.

        Parameters
        ----------
        serialized_string: `str`
            The serialized string.

        Returns
        -------
        obj: any
            The deserialized object.
        """
        return dill.loads(codecs.decode(serialized_string.encode(), "base64"))

    def _get_keys_from_serialized_dict(self, serialized_dict):
        """Returns the keys from a `serialized_dict`, the output of `dumps`.

        Parameters
        ----------
        serialized_dict: `dict`
            The serialized dictionary.

        Returns
        -------
        keys: `tuple`
            A tuple of (pickles, directories, obj_types).
                - pickles: A list of keys that ends with ".pkl".
                - directories: A list of keys that ends with ".dir".
                - obj_types: A dictionary of {key: obj_type} for keys that ends with ".type".
        """
        serialized_keys = list(serialized_dict.keys())
        pickles = [key for key in serialized_keys if PICKLE_KEY_EXTENSION in key]
        directories = [key for key in serialized_keys if DIRECTORY_KEY_EXTENSION in key]
        obj_types = {key.split(".")[0]: self.loads_from_str(serialized_dict[key]) for key in
                     serialized_keys if TYPE_KEY_EXTENSION in key}
        # There should be one dir_key for every type_key
        if not all([directory.split(".")[0] in obj_types for directory in directories]):
            raise ValueError("type and directories do not match.")

        return pickles, directories, obj_types
