# xsimplot

A python script for plotting the spectra calculated with the
[xsim](https://cfour.uni-mainz.de/cfour/index.php) program.

## Usage

For help see
```bash
xsimplot.py --help
```
All command line options can be saved as an `xsimplot.toml` file which
`xsimplot.py` will read by default. You can name the config file something else
but then you need to give the name to the script using the `-c` option.

If an option is specified both in the config file and on the command line, the
command line value is used.

### Options available only from the config file

A list of reference peaks was made to enable simple comparison with
experimental spectra. 
```toml
[[reference_peaks]]
energy = 1.0  # float
energy_unit = "eV"  # currently supporting only eV
amplitude = 1.0  # float
assignment = "$1 ^0 _1$"  # str
```
To add multiple peaks add one entry like that for each peak, see [array of
tables in TOML](https://toml.io/en/v1.0.0#array-of-tables).

## Dependencies 

### Extras
[matplotlib](https://matplotlib.org/)

[numpy](https://numpy.org/)


### Standard python libraries:
[argparse](https://docs.python.org/3/library/argparse.html)

[os](https://docs.python.org/3/library/os.html)

[sys](https://docs.python.org/3/library/sys.html)

[math](https://docs.python.org/3/library/math.html)

[tomllib](https://docs.python.org/3/library/tomllib.html)

### Optional

cofur parsers: xsim

Parsing xsim output gives more information: allows to print basis set
information with the `--verbose` flag. If not wanted use the `--no_parser` flag
and use the `fort.20` files instead of xsim's output.

