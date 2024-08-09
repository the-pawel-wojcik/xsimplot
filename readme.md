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


#### Add reference peaks from a file
Reference peaks (e.g. for an experiment vs theory showcase) can be added from
an external csv file. The file must have two columns: the first column lists
energies while the second column lists intensities of the features.  The file
must have a header `energy,intensity` as its first line. Third optional column
can contain assignment strings. Not all peaks need to have an assignmnet
string.

An example of a file with reference peaks:
```csv
energy,intensity,assignment
30339,2,10a$_1^1$
30515,2,16b$_2^2$
30635,3,5$_1^1$
30695,15,16b$^1_1$
30745,2,6a$^0_1$16b$^2_0$
30876,100,origin
30934,23,16a$^1_1$
31259,17,10a$^1_0$
31277,13,6a$^1_0$16b$^1_1$
31343,12,16b$^2_0$
```

The above cvs file can be added to the spectrum using the following section in
the config
```toml
[[reference_spectrum]]
file = "/home/data/reference_peaks.csv" # absolute paths work the best
plot_type = "stems" # chose "scatter" for experimental band shapes
energy_units = "cm-1" # defaults to "eV"
rescale_intensities = 10.0 # defaults to 1.0, i.e., intensities remain unchanged
```

#### Add reference peaks directly to the spectrum
A list of reference peaks was made to enable simple comparison with
experimental spectra. 
```toml
[[reference_peaks]]
energy = 1.0 # float
energy_unit = "eV" # currently supporting only "eV" and "cm-1"
amplitude = 1.0 # float
assignment = "$1 ^0 _1$" # str
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

