# xsimplot

A python script for plotting the spectra calculated with the [xsim](https://cfour.uni-mainz.de/cfour/index.php) program.

## Usage

For help see
```bash
xsimplot.py --help
```

## Dependencies 

### Extras
[matplotlib](https://matplotlib.org/)

[numpy](https://numpy.org/)


### Standard python libraries:

[argparse](https://docs.python.org/3/library/argparse.html)

[os](https://docs.python.org/3/library/os.html)

[sys](https://docs.python.org/3/library/sys.html)

[math](https://docs.python.org/3/library/math.html)

### Optional

cofur parsers: xsim

Parsing xsim output gives more information: allows to print basis set
information with the `--verbose` flag. If not wanted use the `--no_parser` flag
and use the `fort.20` files instead of xsim's output.

