# xsimplot
A python script for plotting the vibronic spectra. It offers the best support
for outputs of the
[xsim](https://cfour.uni-mainz.de/cfour/index.php),
[ezFCF](https://github.com/the-pawel-wojcik/ezFCF), and
[xfc2](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Franck-CondonInterface)
programs, as well as for a general spectrum in a "ref" file format (see below).

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

#### Add reference peaks from a file
Reference peaks (e.g. for an experiment vs theory showcase) can be added from
an external csv file. The file must have two columns: the first column lists
energies while the second column lists intensities of the features. The file
must have a header `energy,intensity` as its first line. Third, optional,
column can contain assignment strings. Not all peaks need to have the
assignment string.

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
spectrum_files = ["/home/data/reference_peaks.csv"] # absolute paths are best
energy_units = "cm-1" # defaults to "eV"
rescale_intensities = -5.0 # defaults to 1.0, i.e., intensities remain unchanged
y_offset = 1.0 # defaults to 0.0, beginning of the reference peaks
match_origin = 2.71  # in the same units as `energy_units`
line_kwargs.color = 'tab:orange'  # Specify any kwargs for the plot function
```

## Dependencies 

### Extras
[matplotlib](https://matplotlib.org/)

[numpy](https://numpy.org/)

[adjustText](https://github.com/Phlya/adjustText)

### Standard python libraries:
[argparse](https://docs.python.org/3/library/argparse.html)

[os](https://docs.python.org/3/library/os.html)

[sys](https://docs.python.org/3/library/sys.html)

[math](https://docs.python.org/3/library/math.html)

[tomllib](https://docs.python.org/3/library/tomllib.html)

### Optional

If using `--spectrum_format 'xsim'` the additional dependency is needed [cofur
parser](https://github.com/the-pawel-wojcik/cfour_parser). xsim outputs can be
plotted without it if you chose the `--spectrum_format 'fort.20'` instead.

Parsing xsim allows to show basis set and number of Lanczos iterations using
the `--verbose` flag. 

