
import argparse
import json
import h5py
from collections import Counter
import sys
from keras_layer_converters import layer_converters, skip_layers

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.arch_file, 'r') as arch_file:
        arch = json.load(arch_file)
    with open(args.variables_file, 'r') as inputs_file:
        inputs = json.load(inputs_file)

    _check_version(arch)
    #if arch["class_name"] != "Sequential":
    #    sys.exit("this is not a Sequential model, try using kerasfunc2json")

    with h5py.File(args.hdf5_file, 'r') as h5:
        out_dict = {
            'layers': _get_layers(arch, inputs, h5),
        }
        out_dict.update(_parse_inputs(inputs))
    print(json.dumps(out_dict, indent=2, sort_keys=True))

def _check_version(arch):
    if 'keras_version' not in arch:
        sys.stderr.write(
            'WARNING: no version number found for this archetecture!\n')
        return
    #major, minor,_ = arch['keras_version'].split('.')
    #if major != '1' or minor < '2':
    #    warn_tmp = (
    #        "WARNNING: This converter was developed for Keras version 1.2. "
    #        "Your version (v{}.{}) may be incompatible.\n")
    #    sys.stderr.write(warn_tmp.format(major, minor))

def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from Keras saved NN to JSON",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('arch_file', help='architecture json file')
    parser.add_argument('variables_file', help='variable spec as json')
    parser.add_argument('hdf5_file', help='Keras weights file')
    return parser.parse_args()


# __________________________________________________________________________
# master layer converter / inputs function

def _get_layers(network, inputs, h5):
    layers = []
    print(network)
    
    in_layers = network['config']
    in_layers = in_layers['layers']
    print(in_layers)
    n_out = len(inputs['inputs'])
    for layer_n in range(len(in_layers)):
        if not layer_n: continue
        # get converter for this layer
        layer_arch = in_layers[layer_n]
        layer_type = layer_arch['class_name'].lower()
        if layer_type in skip_layers: continue
        convert = layer_converters[layer_type]

        # build the out layer
        out_layer, n_out = convert(h5, layer_arch['config'], n_out)
        layers.append(out_layer)
    return layers

def _parse_inputs(keras_dict):
    inputs = []
    defaults = {}
    for val in keras_dict['inputs']:
        inputs.append({x: val[x] for x in ['offset', 'scale', 'name']})

        # maybe fill default
        default = val.get("default")
        if default is not None:
            defaults[val['name']] = default
    out = {
        'inputs': inputs,
        'outputs': keras_dict['class_labels'],
        'defaults': defaults,
    }
    if 'miscellaneous' in keras_dict:
        misc_dict = {}
        for key, val in keras_dict['miscellaneous'].items():
            misc_dict[str(key)] = str(val)
        out['miscellaneous'] = misc_dict
    return out

if __name__ == '__main__':
    _run()
