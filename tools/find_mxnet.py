from __future__ import print_function
import os
try:
    if os.environ.get('MXNET_EXAMPLE_SSD_DISABLE_PRE_INSTALLED', 0):
        raise ImportError
    import mxnet as mx
    print("Using mxnet as:")
    print(mx)
    print("Warning: using pre-installed version of mxnet may cause unexpected error...")
    print("(export MXNET_EXAMPLE_SSD_DISABLE_PRE_INSTALLED=1) to prevent loading pre-installed mxnet.")
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(curr_path, "../mxnet/python"))
    import mxnet as mx
