#!/usr/bin/env python

import argparse
import csv
import os
import sys
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
import tomllib
from adjustText import adjust_text


DISREGARD_INTENSITY = 1e-20  # From xsim's spectrum
ANNOTATION_DISREGARD_THRESH = 0.01  # as a part of the spectrum tallest peak

FONTSIZE = 12
CM2INCH = 1/2.54

CM2eV = 1./8065.543937
eV2CM = 8065.543937

supported_units = {
    "eV": lambda x: x,
    "cm-1": lambda x: x * CM2eV,
    "nm": lambda x: 1239.84198 / x,
}

SUPPORTED_FILETYPES = ["xsim", "fort.20", "ezFCF", 'ref']
SUPPORTED_ENVELOPES = ['stack', 'overlay', 'sum only']
ALLOWED_ANNOTATION_POSITIONS = [
    "top left", "top center", "top right",
    "center left", "center center", "center right",
    "bottom left", "bottom center", "bottom right",
]
ALLOWED_SECOND_AXIS_STYLES = ["cm", "cm offset", "nm"]


class Spectrum:
    """Representation of the spectrum as a list of `spectral_points`. Each
    spectral point is a dictionary of
    `{"energy": float, "intenity": float, "annotation": str}`
    where "annotation" is optional. Look at the value of
    `assignments_available` to tell if the annotations are available. The
    "energy" is saved in eV.
    """

    def __init__(self):
        self.spectral_points = list()
        self.energy_unit: str = "eV"
        self.assignments_available: str = None
        self.intentities_are_positive: bool = True

    def add_sepctral_point(
            self,
            energy: float,
            intensity: float,
            assignment: str = None,
    ):
        if assignment is None:
            self.spectral_points.append({
                'energy': energy,
                'intensity': intensity,
            })
        else:
            self.spectral_points.append({
                'energy': energy,
                'intensity': intensity,
                'assignment': assignment,
            })

    def update_assignments_flag(self):
        """ Check if spectral points have the `assignment` value set. The flag
        values are None, "all" or "some". """
        number_annotations = 0
        for point in self.spectral_points:
            if 'assignment' in point:
                number_annotations += 1
        if number_annotations == 0:
            self.assignments_available = None
        elif number_annotations == len(self.spectral_points):
            self.assignments_available = "all"
        else:
            self.assignments_available = "some"

    def update_intensities_sign_flag(self):
        """ Check if spectral points show positive intensities.
        Check only for peaks with assignment available. """
        if self.assignments_available is None:
            self.intentities_are_positive = True
            return

        last_sign = None
        for point in self.spectral_points:
            if 'assignment' not in point:
                continue
            if last_sign is None:
                last_sign = point['intensity'] > 0
                continue
            if (point['intensity'] > 0) != last_sign:
                print("Warning: some intensities are positieve and some are"
                      " negative. Treating all as positive.", file=sys.stderr)
                self.intentities_are_positive = True
                return
        self.intentities_are_positive = last_sign

    def get_origin_energy(self):
        """ Position of the lowest energy peak. """
        self.spectral_points.sort(key=lambda x: x['energy'])
        return self.spectral_points[0]['energy']

    def get_max_peak_energy(self):
        """ Position of the highest energy peak. """
        self.spectral_points.sort(key=lambda x: x['energy'])
        return self.spectral_points[-1]['energy']

    def get_max_intensity(self):
        """Returns max of abs intensity."""
        max_peak = max(self.spectral_points, key=lambda x: abs(x['intensity']))
        return max_peak['intensity']

    def set_all_intensities_even(self):
        for peak in self.spectral_points:
            peak['intensity'] = 1.0

    def __add__(self, other):
        if isinstance(other, float):
            for peak in self.spectral_points:
                peak['energy'] += other
        else:
            raise NotImplementedError
        return self

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, float):
            for peak in self.spectral_points:
                peak['intensity'] *= other
        else:
            raise NotImplementedError
        return self


class SpectrumTweaks:
    """ Access common spectrum tweaks:
        ```python
        shift_eV: float
        uniform_intensities: bool
        rescale_factor: float
        ```
    """

    def __init__(
        self,
        shift_eV: float = 0.0,
        uniform_intensities: bool = False,
        rescale_factor: float | None = None,
    ):
        self.shift_eV = shift_eV
        self.uniform_intensities = uniform_intensities
        self.rescale_factor = None

    def set_shift_eV(self, shift_eV: float):
        self.shift_eV = shift_eV

    def set_uniform_intensities(self, value: bool):
        if isinstance(value, bool):
            self.uniform_intensities = value
        else:
            raise ValueError

    def set_rescale_intensities(self, rescale_factor: float):
        self.rescale_factor = rescale_factor

    def apply_to(self, spectrum: Spectrum):
        if self.rescale_factor is not None:
            spectrum *= self.rescale_factor

        if self.uniform_intensities is True:
            spectrum.set_all_intensities_even()

        if self.shift_eV is not None:
            for peak in spectrum.spectral_points:
                peak['energy'] -= self.shift_eV


def ezFCF_label_helper(
        state: str,
) -> str:
    """
    turn st_number(n1vm1,n2vm2,...) into {m1: n1, m2: n2, ...}
    """
    vibrational_state = state.split('(')[1][:-1]  # omg )
    if vibrational_state == "0":
        return {}
    excitations = vibrational_state.split(',')
    out = {}
    for excitation in excitations:
        n_quanta, n_mode = excitation.split('v')
        out[int(n_mode)] = int(n_quanta)
    return out


def ezFCF_label_to_spectroscopic_label(
        assignment: str,
        first_is_the_lowest: bool = True,
) -> str:
    initial, final = assignment.split('->')
    initial = ezFCF_label_helper(initial)
    final = ezFCF_label_helper(final)
    active_modes = set(initial.keys()).union(set(final.keys()))
    active_modes = list(active_modes)
    active_modes.sort()
    if len(active_modes) == 0:
        return "0"

    out_str = ""
    for mode in active_modes:
        quanta_initial = 0
        if mode in initial:
            quanta_initial = initial[mode]
        quanta_final = 0
        if mode in final:
            quanta_final = final[mode]

        if first_is_the_lowest is True:
            up = quanta_final
            down = quanta_initial
        else:
            up = quanta_initial
            down = quanta_final
        out_str += f"{mode}$^{up}_{down}$"
    return out_str


def parse_command_line() -> argparse.Namespace:
    """Parse arguments from the command line"""
    parser = argparse.ArgumentParser(description="Plot vibronic spectrum.")

    parser.add_argument(
        "--spectrum_files",
        help="List of files storing the spectrum. The -n (--spectrum_format)"
        " flag controls the spectrum format.",
        default=None,
        nargs="+",
    )

    second_ax = parser.add_argument_group(
        'second ax',
    )

    second_ax.add_argument(
        "--second_panel_spectrum_files",
        help="List of files storing the spectrum plotted on a second axis.",
        nargs="+",
        default=None,
    )

    second_ax.add_argument(
        '--second_panel_spectrum_format',
        help='See --spectrum_format',
        default=None,
        type=str,
        choices=SUPPORTED_FILETYPES,
    )

    second_ax.add_argument(
        '--second_panel_energy_units',
        help='See --energy_units',
        default=None,
        type=str,
        choices=supported_units,
    )

    second_ax.add_argument(
        "--second_panel_annotate",
        help="see --annotate",
        default=None,
        type=str,
    )

    second_ax.add_argument(
        '--second_panel_scatter',
        help='See --scatter',
        default=None,
        type=bool,
        action=argparse.BooleanOptionalAction,
    )

    second_ax.add_argument(
        "--second_panel_sticks_off",
        help="See --sticks_off",
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    second_ax.add_argument(
        "--second_panel_match_origin",
        type=float,
        default=None,
        help="See --match_origin"
    )

    second_ax.add_argument(
        "--second_panel_shift_eV",
        help="See --shift_eV",
        type=float
    )

    second_ax.add_argument(
        "--second_panel_rescale_intensities",
        help="See --rescale_intensities",
        type=float,
        default=None
    )

    parser.add_argument(
        "--second_panel_envelope",
        help="See --envelope",
        type=str,
        default=None,
        choices=SUPPORTED_ENVELOPES
    )

    parser.add_argument(
        "--annotate",
        help="Place string with the annotation on the figure."
        "Use 'a' as the first letter of the string to append"
        " the annotation generated by other flags; use 'o' to"
        " overwrite.",
        default=None,
        type=str
    )

    parser.add_argument(
        "--position_annotation",
        help="Place the annotation at",
        choices=ALLOWED_ANNOTATION_POSITIONS,
        default=None,
        type=str
    )

    parser.add_argument("-c", "--config",
                        help="Pick config file. "
                        "If not specified, look for xsimplot.toml.",
                        default="xsimplot.toml")

    parser.add_argument("-e", "--envelope",
                        help="Add a Lorenzian envelope to every peak.",
                        type=str,
                        default=None,
                        choices=SUPPORTED_ENVELOPES)

    parser.add_argument(
        "--gamma",
        help="Gamma in the Lorenzian:\n"
        "(0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma) **2)\n"
        "It also controlls the xlims if not specified otherwise.",
        type=float,
        default=None
    )

    parser.add_argument(
        "--scatter",
        help="Scatter plot the spectrum -- good for experimental data.",
        type=bool,
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--all_intensities_even",
        help="Use this switch to set all intenties to the same value or"
        " to override config's value. See also: `--rescale_intensities`.",
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument("-I", "--rescale_intensities",
                        help="Multiply all intensiteis by the value.",
                        type=float,
                        default=None)

    parser.add_argument("-k", "--horizontal_minor_ticks_2nd_axis",
                        help="Specify the interval at which the minor ticks"
                        " should appear.",
                        type=float,
                        default=None)

    parser.add_argument("-K", "--horizontal_minor_ticks",
                        help="Specify the interval at which the minor ticks"
                        " should appear.",
                        type=float,
                        default=None)

    parser.add_argument(
        "--spectrum_format",
        help="Chose the format of the spectrum file. 'xsim'"
        "requires additional parser. 'ref' marks that the"
        " input follows the 'reference_spectrum' format"
        " described in the reamde file. If 'ref' uses peak "
        " location not in 'eV', see '-u'.",
        default=None,  # defaults to fort.20 see code
        type=str,
        choices=SUPPORTED_FILETYPES,
    )

    parser.add_argument("-r", "--scale_factor",
                        help="Scale the figure size with the factor.",
                        default=None,
                        type=float)

    parser.add_argument(
        "--sticks_off",
        help="Switch allowing to hide the stick spectrum.",
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--energy_units",
        help="For use with '-n ref';"
        " allows to change energy units of the spectrum"
        " (defaults to eV).",
        default=None,  # defaults to eV
        type=str,
        choices=supported_units,
    )

    parser.add_argument(
        "--y_offset",
        help="Shift the spectrum up or down.",
        type=float,
        default=None,
    )

    parser.add_argument(
        "-v", "--verbose",
        help="Annotate the plot with # of Lanczos iterations and basis size.",
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    save = parser.add_mutually_exclusive_group()

    save.add_argument("-d", "--dont_save",
                      help="Just show the figure. Don't save it yet.",
                      default=None,
                      action="store_true")

    save.add_argument("-f", "--filename",
                      help="Save figure as filename.",
                      default=None)

    shift = parser.add_mutually_exclusive_group()

    help = "Shift simulated spectrum to align the first peak at <match_origin>"
    help += " eV."
    shift.add_argument("-o", "--match_origin",
                       type=float,
                       default=None,
                       help=help)

    shift.add_argument("-s", "--shift_eV",
                       help="Positive shift moves peak towards lower energy.",
                       type=float)

    parser.add_argument("-x", "--second_axis",
                        help="Add a second energy axis; pick units. If you"
                        " want to see the offset from the first peak add"
                        " 'offset'.",
                        type=str,
                        choices=ALLOWED_SECOND_AXIS_STYLES,
                        default=None)

    parser.add_argument(
        "--show_yaxis_ticks",
        help="Switch for displaying ticks on the yaxis.",
        default=None,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    return args


def height_one_lorenzian(x: float, x0: float, gamma: float) -> float:
    return lorenzian(x, x0, gamma) * m.pi * 0.5 * gamma


def lorenzian(x: float, x0: float, gamma: float) -> float:
    '''
    This definition of a Lorenzian comes from Eq. 2.40

    Köppel, H., W. Domcke, and L. S. Cederbaum. “Multimode Molecular Dynamics
    Beyond the Born-Oppenheimer Approximation.” In Advances in Chemical Physics
    edited by I. Prigogine and Stuart A. Rice, LVII:59–246, 1984.
    https://doi.org/10.1002/9780470142813.ch2.

    gamma corresponds to full width at half max (FWHM)
    '''
    return 1.0 / m.pi * (0.5 * gamma / ((x - x0)**2 + (0.5 * gamma) ** 2))


def turn_spectrum_to_xs_and_ys(
        spectrum: list[dict],
) -> tuple[float, float]:
    xs = []
    ys = []
    for spectral_point in spectrum:
        xs += [spectral_point['Energy (eV)']]
        ys += [spectral_point['Relative intensity']]

    return xs, ys


def lorenz_intensity(
        x: float,
        gamma: float,
        spectrum: Spectrum,
) -> float:
    out = 0.0
    for spectral_point in spectrum.spectral_points:
        energy_eV = spectral_point['energy']
        intensity = spectral_point['intensity']
        out += intensity * height_one_lorenzian(x, energy_eV, gamma)
    return out


def stem_spectral_peaks(
        peaks: list[dict],
        ax: mpl.axes.Axes,
        y_offset: float = 0.0,
        **line_kwargs
):

    for spectral_point in peaks:
        energy_eV = spectral_point['energy']
        intensity = spectral_point['intensity']
        # ax.vlines(x=energy_eV, ymin=y_offset, ymax=intensity, colors=color)
        ax.vlines(
            x=energy_eV,
            ymin=y_offset,
            ymax=intensity+y_offset,
            **line_kwargs
        )


def get_xsim_outputs_from_fort20(
        spectrum_files: list[str],
) -> tuple[list[Spectrum], int]:
    xsim_outputs = []
    lanczos = None
    for file_idx, fort20_file in enumerate(spectrum_files):

        with open(fort20_file, 'r') as f:
            fort20 = f.readlines()

        loc_lanczos = int(fort20[0].split()[0])
        if lanczos is None:
            lanczos = loc_lanczos
        elif loc_lanczos != lanczos:
            print("Warning: the number of Lanczos iterations in the input"
                  f" file #{file_idx} differs from the previous files.",
                  file=sys.stderr)

        xsim_spectrum = Spectrum()
        for peak in fort20[1:]:
            peak = peak.split()
            try:
                intensity = float(peak[1])
            except ValueError:
                # for tiny tiny intensities xsim produces garbage, i.e.:
                # 6.86644579832093   0.159775868164848-105  39294.6062024455
                exponent = int(peak[1][-4:])
                if exponent < -99:
                    continue
                print("Error: cannot process the line:" + peak,
                      file=sys.stderr)
                sys.exit(1)
            if intensity < DISREGARD_INTENSITY:
                continue

            energy_eV = float(peak[0])
            xsim_spectrum.add_sepctral_point(
                energy=energy_eV,
                intensity=intensity,
            )

        xsim_outputs += [xsim_spectrum]

    return xsim_outputs, lanczos


def get_xsim_outputs(
        spectrum_files: list[str],
) -> tuple[list[Spectrum], str, int]:

    from cfour_parser.xsim import parse_xsim_output
    xsim_outputs = []
    basis = None
    lanczos = None
    for file_idx, spectrum_file in enumerate(spectrum_files):
        with open(spectrum_file) as f:
            xsim_data = parse_xsim_output(f)

        xsim_spectrum = Spectrum()
        for data_point in xsim_data['spectrum_data']:
            xsim_spectrum.add_sepctral_point(
                energy=data_point['Energy (eV)'],
                intensity=data_point['Relative intensity'],
            )

        xsim_outputs.append(xsim_spectrum)

        loc_basis = xsim_data['basis']
        if basis is None:
            basis = loc_basis
        elif basis != loc_basis:
            print("Warning: Outputs use different basis sets.",
                  file=sys.stderr)

        loc_lanczos = int(xsim_data['Lanczos'])
        if lanczos is None:
            lanczos = loc_lanczos
        elif lanczos != loc_lanczos:
            print("Warning! Outputs use different # of Lanczos iterations.",
                  file=sys.stderr)

    return xsim_outputs, basis, lanczos


def parse_ref_spectrum(
        filename: str,
        convert_to_eV
) -> Spectrum:
    ref_spectrum = Spectrum()
    assignments_are_available = False
    filename = os.path.expanduser(filename)
    with open(filename, 'r', newline='') as spectrum_file:
        reader = csv.DictReader(spectrum_file)
        assignments_are_available = 'assignment' in reader.fieldnames
        if assignments_are_available:
            for row in reader:
                energy_input = float(row['energy'])
                energy_eV = convert_to_eV(energy_input)
                ref_spectrum.add_sepctral_point(
                    energy=energy_eV,
                    intensity=float(row['intensity']),
                    assignment=row['assignment'],
                )
        else:
            for row in reader:
                energy_input = float(row['energy'])
                energy_eV = convert_to_eV(energy_input)
                ref_spectrum.add_sepctral_point(
                    energy=energy_eV,
                    intensity=float(row['intensity']),
                )

    return ref_spectrum


def find_energy_units(
        config: dict = {},
        args: argparse.Namespace | None = None,
) -> str:
    energy_units = None
    if 'energy_units' in config:
        if config['energy_units'] not in supported_units:
            print("Error: the config file requests unsupported energy units:\n"
                  f"\tenergy_units = {config['energy_units']}",
                  file=sys.stderr)
            sys.exit(1)
        energy_units = config['energy_units']
    if args is not None and args.energy_units is not None:
        energy_units = args.energy_units
    if energy_units is None:
        energy_units = 'eV'
    return energy_units


def get_ref_spectrum(
        spectrum_files: list[str],
        energy_units: str = "eV",
) -> list[Spectrum]:
    """
    Ref spectrum means spectrum as csv file.
    energy,intensity,assignmnet
    assignment is optional
    """
    convert_to_eV = supported_units[energy_units]

    spectra = []
    for file_idx, spectrum_file in enumerate(spectrum_files):
        ref_spectrum = parse_ref_spectrum(spectrum_file, convert_to_eV)
        spectra += [ref_spectrum]

    return spectra


def get_ezFCF_spectrum(
        spectrum_files: list[str],
) -> list[Spectrum]:
    spectra = []
    for file_idx, spectrum_file in enumerate(spectrum_files):
        with open(spectrum_file) as f:
            lines = f.readlines()

        ezFCF_spectrum = Spectrum()
        for line in lines:
            splitline = line.split()
            energy_eV = float(splitline[0])
            intensity = float(splitline[1])
            ezfcf_assignment = splitline[4]
            assignment = ezFCF_label_to_spectroscopic_label(ezfcf_assignment)
            ezFCF_spectrum.add_sepctral_point(
                energy=energy_eV,
                intensity=intensity,
                assignment=assignment,
            )
        spectra.append(ezFCF_spectrum)

    return spectra


def find_shift_eV(
        config: dict = {},
        args: argparse.Namespace | None = None,
) -> float | None:
    shift_eV = None
    if 'shift_eV' in config:
        shift_eV = config['shift_eV']
    if args is not None and args.shift_eV is not None:
        shift_eV = args.shift_eV

    return shift_eV


def find_match_origin(
        config: dict = {},
        args: argparse.Namespace | None = None,
) -> float | None:
    match_origin = None
    if 'match_origin' in config:
        match_origin = config['match_origin']
    if args is not None and args.match_origin is not None:
        match_origin = args.match_origin

    return match_origin


def calculate_spectrum_shift(
        spectra: list[Spectrum],
        shift_eV: float | None,
        match_origin: float | None,
) -> float:
    """
    Find shift (in eV) which will be applied to the spectrum.
    """

    if shift_eV is not None:
        return shift_eV

    if match_origin is not None:
        origins = []
        for spectrum in spectra:
            local_orign = spectrum.get_origin_energy()
            origins.append(local_orign)
        origin_eV = min(origins)
        shift_eV = origin_eV - match_origin
        return shift_eV

    return None


def calculate_xlims(
        spectra: list[Spectrum],
        gamma_eV: float,
        how_far: int = 4,
) -> tuple[float, float]:
    """
    The xlims are how_far*gamma away from the first and last peak.
    TODO: this might be too much for some narrower spectra. When there is no
    envelope specified you should leave the default ones.
    """

    # Default values
    mins = list()
    maxes = list()

    for spectrum in spectra:
        mins.append(spectrum.get_origin_energy())
        maxes.append(spectrum.get_max_peak_energy())

    minimum = min(mins)
    maximum = max(maxes)
    left = minimum - how_far * gamma_eV
    right = maximum + how_far * gamma_eV

    return left, right


def add_scatter(
        ax: mpl.axes.Axes,
        spectra: list[Spectrum],
        scatter: bool,
        y_offset: float,
) -> float:
    """ Returns max value of the scatter. """
    if scatter is False:
        return 0.0

    max_values = list()
    for spectrum in spectra:
        xs = [peak['energy'] for peak in spectrum.spectral_points]
        ys = [
            peak['intensity'] + y_offset for peak in spectrum.spectral_points
        ]
        ax.plot(xs, ys, marker='o', markersize=2, linewidth=0.5)
        max_values.append(max(ys))

    return max(max_values) - y_offset


def add_envelope(
        ax: mpl.axes.Axes,
        envelope_type: str,
        spectra: list[Spectrum],
        xlims: list[float],
        gamma: float | None,
        y_offset: float,
) -> float:
    """ Returns height (from y_offset to the top) of the tallest feature. """
    if envelope_type is None:
        return 0.0

    npoints = 1500
    xs = np.linspace(xlims[0], xlims[1], npoints)
    accumutated_ys = np.zeros_like(xs) + y_offset

    for spectrum_idx, spectrum in enumerate(spectra):
        state_spectrum = [lorenz_intensity(x, gamma, spectrum) for x in xs]
        state_spectrum = np.array(state_spectrum)
        if envelope_type == "stack":
            ax.fill_between(xs, accumutated_ys + state_spectrum,
                            accumutated_ys, alpha=0.2)
        elif envelope_type == "overlay":
            ax.fill_between(xs, state_spectrum, np.zeros_like(xs), alpha=0.2)
        accumutated_ys += state_spectrum

    if envelope_type in ["overlay", "sum only"]:
        ax.plot(xs, accumutated_ys, color='tab:gray', lw=1)

    fig_max_y = np.max(accumutated_ys)

    return fig_max_y - y_offset


def add_peaks(
        ax: mpl.axes.Axes,
        sticks_off: bool,
        spectra: list[Spectrum],
        xlims: list[float],
        y_offset: float,
        line_kwargs: dict = {},
) -> float:
    """ Returns the height of the tallest added peak. """
    if sticks_off is True:
        return 0.0

    tcolors = list(mcolors.TABLEAU_COLORS.keys())
    peaks_maxima = []
    for spectrum_idx, spectrum in enumerate(spectra):
        my_peaks = [
            peak for peak in spectrum.spectral_points if
            peak['energy'] >= xlims[0] and peak['energy'] <= xlims[1]
        ]
        if 'color' not in line_kwargs:
            line_kwargs = {
                'colors':  tcolors[spectrum_idx],
            }
        else:
            line_kwargs['colors'] = line_kwargs['color']

        stem_spectral_peaks(my_peaks, ax=ax, y_offset=y_offset, **line_kwargs)

        if len(my_peaks) == 0:
            max_peak = {'intensity': 0.0}
        else:
            max_peak = max(my_peaks, key=lambda x: x['intensity'])

        peaks_maxima.append(max_peak['intensity'])

    return max(peaks_maxima)


def find_annotation_position(
        args: argparse.Namespace,
        config: dict,
) -> dict:

    position = "top left"  # default
    if 'position_annotation' in config:  # allow to set it from the config
        conf_position = config['position_annotation']
        if conf_position not in ALLOWED_ANNOTATION_POSITIONS:
            print(
                "Error: Invalid value for 'position_annotation' in config:\n"
                f"       position_annotation = '{position}'\n"
                "       Allowed values: "
                f"{", ".join(ALLOWED_ANNOTATION_POSITIONS)}",
                file=sys.stderr
            )
        else:
            position = conf_position
    if args.position_annotation is not None:  # command line can overwrite
        position = args.position_annotation

    return position


def add_info_text(
        ax: mpl.axes.Axes,
        position: str = "top left",
        shift_eV: float | None = None,
        basis: str | None = None,
        lanczos: int | None = None,
        envelope_type: str | None = None,
        gamma: float = 0.0,
        annotation: str = "",
        verbose: bool = False,
):

    info_kwargs = {
        'fontsize': FONTSIZE,
        'color': 'k',
        'transform': ax.transAxes,
    }

    list_of_texts: list[str] = list()

    if envelope_type is not None:
        list_of_texts += [r'$\gamma = ' + f'{gamma:.3f}$']

    if shift_eV is not None:
        list_of_texts += [f'$s = {shift_eV:.2f}$ eV']

    if verbose is True:
        if basis is not None:
            list_of_texts += [f'Basis: {basis.split()[0]}']
        if lanczos is not None:
            list_of_texts += [f'Lanczos: {lanczos}']

    text = "\n".join(list_of_texts)

    if annotation != "":
        text_with_newline = "\n".join(annotation[1:].split(r'\n'))
        # append
        if annotation[0] == "a":
            if len(text) == 0:
                text = text_with_newline
            elif len(text_with_newline) != 0:
                text += "\n" + text_with_newline
        # overwrite
        elif annotation[0] == "o":
            text = text_with_newline
        else:
            print("Warning: The annotation texts needs to start with either"
                  " 'a' or 'o', see help for details.", file=sys.stderr)

    x_position = {
        "top left": 0.025,
        "top center": 0.5,
        "top right": 0.975,
        "center left": 0.025,
        "center center": 0.5,
        "center right": 0.975,
        "bottom left": 0.025,
        "bottom center": 0.5,
        "bottom right": 0.975,
    }

    y_position = {
        "top left": 0.9,
        "top center": 0.9,
        "top right": 0.9,
        "center left": 0.5,
        "center center": 0.5,
        "center right": 0.5,
        "bottom left": 0.05,
        "bottom center": 0.05,
        "bottom right": 0.05,
    }

    info_kwargs.update({
        'horizontalalignment': position.split()[1],
        'verticalalignment': position.split()[0],
    })

    ax.text(x_position[position], y_position[position], text, **info_kwargs,
            bbox=dict(facecolor='white'))


def prepare_filename(
        spectrum_files: list[str],
) -> str:

    filename = str()
    for idx, outname in enumerate(spectrum_files):
        if idx > 0:
            filename += "+"
        filename += os.path.basename(outname)
    filename += '_spectrum.pdf'

    return filename


def get_origin_eV(spectra: list[Spectrum]) -> float:
    origins = list()
    for spectrum in spectra:
        origins.append(spectrum.get_origin_energy())

    return min(origins)


def find_minor_ticks_interval(
        args: argparse.Namespace | None,
        config: dict,
) -> float:
    interval = None
    if 'horizontal_minor_ticks' in config:
        interval = config['horizontal_minor_ticks']
    if args is not None and args.horizontal_minor_ticks is not None:
        interval = args.horizontal_minor_ticks

    return interval


def add_second_axis(
        ax: mpl.axes.Axes,
        origin_eV: float,
        second_axis: str | None,
        interval: float | None,
) -> mpl.axes.Axes | None:

    if second_axis is None:
        return None

    if second_axis == "cm offset":
        second_ax: mpl.axes.Axes = add_cm_scale(
            ax,
            interval=interval,
            origin_eV=origin_eV
        )

    elif second_axis == "cm":
        second_ax: mpl.axes.Axes = add_cm_scale(ax, interval=interval)

    elif second_axis == "nm":
        second_ax: mpl.axes.Axes = add_nm_scale(ax)

    return second_ax


def add_cm_scale(
        ax: mpl.axes.Axes,
        interval: float | None = None,
        origin_eV: float | None = None,
) -> mpl.axes.Axes:
    ax_cm: mpl.axes.Axes = ax.twiny()

    if origin_eV is None:
        origin_cm = 0
    else:
        origin_cm = origin_eV * eV2CM

    x1, x2 = ax.get_xlim()
    x1 = x1 * eV2CM - origin_cm
    x2 = x2 * eV2CM - origin_cm

    ax_cm.set_xlim(x1, x2)

    if interval is not None:
        ax_cm.xaxis.set_minor_locator(MultipleLocator(interval))

    return ax_cm


def add_nm_scale(
        ax: mpl.axes.Axes,
        interval: float | None,
) -> mpl.axes.Axes:
    r""" Relation between photon's energy and wavelength:
        E = hc / \lambda
    """
    ToLambda = 1239.84198  # from eV to nm
    ax_nm = ax.twiny()

    x1, x2 = ax.get_xlim()
    x1 = ToLambda / x1
    x2 = ToLambda / x2

    # TODO: I don't think this works. I think that you leave the scale linear
    # in energy. Is this good for the ticks location?
    ax_nm.set_xlim(x1, x2)

    if interval is not None:
        ax_nm.xaxis.set_minor_locator(MultipleLocator(interval))

    return ax_nm


def get_fig_and_ax(
        args: argparse.Namespace,
        config: dict,
) -> tuple[mpl.figure.Figure, mpl.axis.Axis, mpl.axis.Axis | None]:

    # Scale factor is used to resize the figure in both direction
    # Making the figures smaller is the same as making the text larger
    scale_factor = 1.0
    if 'scale_factor' in config:
        scale_factor = config['scale_factor']
    if args.scale_factor is not None:
        scale_factor = args.scale_factor

    # Aspect ratio = width/height
    # width = height * ar
    aspect_ratio = 16.0/9.0
    if 'aspect_ratio' in config:
        aspect_ratio = config['aspect_ratio']

    # The default figure size is 12cm x 12 cm. Smaller should be better.
    FIGSIZE = 10 * CM2INCH * scale_factor
    second_panel_spectrum_files = find_value_of(
        'second_panel_spectrum_files',
        config,
        args,
        default=None,
    )

    if second_panel_spectrum_files is not None:
        fig, ax = plt.subplots(
            nrows=2,
            figsize=(FIGSIZE * aspect_ratio, FIGSIZE),
            layout='constrained'
        )
        return fig, ax[0], ax[1]
    else:
        fig, ax = plt.subplots(
            figsize=(FIGSIZE * aspect_ratio, FIGSIZE),
            layout='constrained'
        )
        return fig, ax, None


def add_spectrum_assignments(
        ax: mpl.axes.Axes,
        spectra: list[Spectrum],
        top_feature: float,
        xlims: list[float],
        y_offset: float = 0.0,
) -> list[mpl.text.Text]:

    text_kwargs = {
        'ha': 'center',
        'va': 'bottom',
    }

    texts = list()
    for spectrum in spectra:
        spectrum.update_assignments_flag()
        if spectrum.assignments_available is None:
            continue

        spectrum.update_intensities_sign_flag()
        text_kwargs = {
            'ha': 'center',
        }
        if spectrum.intentities_are_positive is True:
            text_kwargs['va'] = 'bottom'
        else:
            text_kwargs['va'] = 'top'

        for peak in spectrum.spectral_points:
            if 'assignment' not in peak:
                continue
            energy_eV = peak['energy']
            amplitude = peak['intensity']
            assignment = peak['assignment']

            # Do not draw annotations for peaks which are not displayed
            if energy_eV > xlims[1] or energy_eV < xlims[0]:
                continue

            # Disregard small features
            if amplitude < ANNOTATION_DISREGARD_THRESH * top_feature:
                continue

            text_obj = ax.text(
                energy_eV,
                amplitude + y_offset,
                assignment,
                **text_kwargs
            )
            texts.append(text_obj)

    return texts


def get_config(args: argparse.Namespace) -> dict:
    config_file = os.path.expanduser(args.config)
    if os.path.isfile(config_file) is False:
        print(f"Info: No config file {args.config} present.", file=sys.stderr)
        return {}

    print(f"Info: Using the config file {args.config}.", file=sys.stderr)
    with open(config_file, 'rb') as config_toml:
        config = tomllib.load(config_toml)

    return config


def find_if_show_yaxis_ticks(
        args: argparse.Namespace | None,
        config: dict,
) -> bool:
    show_yaxis_ticks = None
    if 'show_yaxis_ticks' in config:
        show_yaxis_ticks = config['show_yaxis_ticks']
    if args is not None and args.show_yaxis_ticks is not None:
        show_yaxis_ticks = args.show_yaxis_ticks

    if show_yaxis_ticks is None:
        show_yaxis_ticks = False  # set the default
    return show_yaxis_ticks


def find_ylims(
        config: dict,
):
    """Returns a dictionary with
    ```python
    {'bottom': float, 'top': float}
    ```"""
    if 'ylims' in config:
        ylims: dict = config['ylims']
        if not isinstance(ylims, dict):
            print("Warning: The conifg 'ylims' is invalid", file=sys.stderr)
            return None
        if 'bottom' not in ylims or 'top' not in ylims:
            print("Warning: The conifg 'ylims' is invalid", file=sys.stderr)
            return None

        for val in ylims.values():
            if not isinstance(val, float):
                print("Warning: The conifg specifies invalid 'ylims'.",
                      file=sys.stderr)
                return None
        return ylims
    return None


def find_spectrum_format(
        config: dict,
        args: argparse.Namespace | None = None,
) -> str:
    spectrum_format = None
    if 'spectrum_format' in config:
        if config['spectrum_format'] not in SUPPORTED_FILETYPES:
            print("Error: the config file contains invalid spectrum format\n"
                  f"\tspectrum_format = {config['spectrum_format']}",
                  file=sys.stderr)
            sys.exit(1)
        spectrum_format = config['spectrum_format']

    # The command line can override it
    if args is not None and args.spectrum_format is not None:
        spectrum_format = args.spectrum_format
    # if not specified it defaults to fort.20
    if spectrum_format is None:
        spectrum_format = "fort.20"

    return spectrum_format


def find_value_of(
        property: str,
        config: dict = {},
        args: argparse.Namespace | None = None,
        property_type=None,
        allowed_values: list = None,
        default=None,
):
    value = None

    # Look for it in config
    if property in config:
        value = config[property]
        if allowed_values is not None and value not in allowed_values:
            print(
                f"Error: config specifies invalid value for {property}.\n"
                f"\t{property} = {value}"
                f"\tAllowed values: {" ".join(allowed_values)}",
                file=sys.stderr
            )
            return default
        if property_type is not None and not isinstance(value, property_type):
            print(
                f"Error: config specifies invalid value for {property}.\n"
                f"\t{property} = {value}"
                f"\tProperty must be of type: {property_type}",
                file=sys.stderr
            )
            return default

    # The command line can override it
    if args is not None and getattr(args, property) is not None:
        value = getattr(args, property)
        # No error checking as argparse is expected to do it

    # if not defined return default
    if value is None:
        value = default

    return value


def collect_spectra(
        spectrum_files: list[str],
        spectrum_format: str,
        energy_units: str,
) -> tuple[list[Spectrum], str, int]:
    """Collect the spectrum specified on the command line."""

    if spectrum_format == "fort.20":
        basis = None
        spectra, lanczos = get_xsim_outputs_from_fort20(spectrum_files)
    elif spectrum_format == "xsim":
        spectra, basis, lanczos = get_xsim_outputs(spectrum_files)
    elif spectrum_format == "ezFCF":
        basis = None
        lanczos = None
        spectra = get_ezFCF_spectrum(spectrum_files)
    elif spectrum_format == "ref":
        basis = None
        lanczos = None
        spectra = get_ref_spectrum(spectrum_files, energy_units)
    else:
        print(f"Error: Unknown spectrum format {spectrum_format}",
              file=sys.stderr)

    return spectra, basis, lanczos


def find_if_uniform_intensities(
        args: argparse.Namespace | None,
        config: dict,
) -> bool:
    """
    Sometimes only the position of peaks is meaningful. The switch
    'all_intensities_even' allows to set all intensities to the same value.
    """
    even_intensites = None
    if 'all_intensities_even' in config:
        even_intensites = config['all_intensities_even']

    if args is not None and args.all_intensities_even is not None:
        even_intensites = args.all_intensities_even

    if even_intensites is None:
        return False

    return even_intensites


def decongest_assignments(
        ax: mpl.axes.Axes,
        texts: list[mpl.text.Text],
        go_up: bool = True,
):
    if len(texts) == 0:
        return

    if go_up is True:
        only_move = {
            "text": "y+",
            "static": "y",
            "explode": "y+",
            "pull": "y",
        }
    else:
        only_move = {
            "text": "y-",
            "static": "y",
            "explode": "y-",
            "pull": "y"
        }

    adjust_text(
        texts,
        ax=ax,
        # objects=objects,
        avoid_self=False,
        only_move=only_move,
        min_arrow_len=20,
        arrowprops=dict(
            arrowstyle='->',
            color='lightgray',
            lw=1,
            mutation_scale=5,
            linestyle='-',
        ),
    )


def collect_ax_tweaks(
        args: argparse.Namespace | None,
        config: dict,
) -> dict:
    spectrum_plot_kw = dict()

    minor_ticks_interval = find_minor_ticks_interval(args, config)
    spectrum_plot_kw['minor_ticks_interval'] = minor_ticks_interval

    show_yaxis_ticks = find_if_show_yaxis_ticks(args, config)
    spectrum_plot_kw['show_yaxis_ticks'] = show_yaxis_ticks

    ylims = find_value_of(
        property='ylims',
        config=config,
        args=None,
        property_type=dict,
        default=None,
    )
    # find_ylims(config)
    spectrum_plot_kw['ylims'] = ylims

    return spectrum_plot_kw


def apply_ax_tweaks(
        ax: mpl.axes.Axes,
        xlims: tuple[float, float],
        ylims: dict[str, float] | None,
        show_yaxis_ticks: bool,
        minor_ticks_interval: float | None,
):
    """
    Parameters:
        ax: matplotlib.axes.Axes = the Axes that will be modified
        ylims: dict[str, float] | None = axis limits; 'bottom' and 'top'.
        show_yaxis_ticks: bool = leave or hide the ticks on the yaxis.
        minor_ticks_interval = add extra minor ticks on axis
    """

    ax.set_xlim([xlims[0], xlims[1]])

    if ylims is not None:
        ax.set_ylim(bottom=ylims['bottom'], top=ylims['top'])

    if show_yaxis_ticks is False:
        ax.set_yticks([])

    if minor_ticks_interval is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks_interval))


def collect_spectrum_tweaks(
        args: argparse.Namespace | None,
        config: dict,
) -> SpectrumTweaks:
    """ Find 'use_uniform_intensities' and 'rescale_factor' values. """
    spectrum_tweaks = SpectrumTweaks()

    use_uniform_intensities = find_if_uniform_intensities(args, config)
    spectrum_tweaks.set_uniform_intensities(use_uniform_intensities)

    rescale_factor = find_value_of(
        property='rescale_intensities',
        config=config,
        args=args,
        property_type=float,
        default=None
    )
    spectrum_tweaks.set_rescale_intensities(rescale_factor)

    return spectrum_tweaks


def plot_spectra(
        ax: mpl.axes.Axes,
        spectra: list[Spectrum],
        xlims: tuple[float, float],
        envelope_type: str | None,
        gamma: float | None,
        sticks_off: bool,
        y_offset: float = 0.0,
        line_kwargs: dict = {},
        scatter: bool = False,
) -> float:
    envelope_max_y = add_envelope(ax, envelope_type, spectra, xlims, gamma,
                                  y_offset)
    max_peak = add_peaks(ax, sticks_off, spectra, xlims, y_offset, line_kwargs)
    scatter_max = add_scatter(
        ax=ax,
        spectra=spectra,
        y_offset=y_offset,
        scatter=scatter
    )
    top_feature = max([envelope_max_y, max_peak, scatter_max])
    return top_feature


def main():
    args = parse_command_line()
    config = get_config(args)
    spectrum_format = find_spectrum_format(config, args)
    spectrum_files = find_value_of(
        property='spectrum_files',
        args=args,
        config=config,
        property_type=list,
        default=[],
    )
    energy_units = find_energy_units(config, args)

    spectra, basis, lanczos = collect_spectra(
        spectrum_files=spectrum_files,
        spectrum_format=spectrum_format,
        energy_units=energy_units,
    )

    annotation_kw = {
        'lanczos': lanczos,
        'basis': basis,
    }

    spectrum_tweaks: SpectrumTweaks = collect_spectrum_tweaks(args, config)

    input_shift_eV = find_shift_eV(config, args)
    match_origin = find_value_of(
        property='match_origin',
        config=config,
        args=args,
        default=None,
        property_type=float | None,
    )
    shift_eV = calculate_spectrum_shift(spectra, input_shift_eV, match_origin)
    spectrum_tweaks.set_shift_eV(shift_eV)
    annotation_kw['shift_eV'] = shift_eV

    for spectrum in spectra:
        spectrum_tweaks.apply_to(spectrum)

    fig, ax, ax2nd = get_fig_and_ax(args, config)

    spectrum_plot_kw = {}

    gamma = find_value_of(
        property='gamma',
        args=args,
        config=config,
        property_type=float,
        default=0.03,
    )
    xlims = find_value_of(
        property='xlims',
        args=None,
        config=config,
        property_type=dict,
        default=None
    )
    # xlims are used as a tuple of two in what comes next
    if xlims is None:
        xlims = calculate_xlims(spectra, gamma)
    else:
        xlims = xlims['left'], xlims['right']

    spectrum_plot_kw['xlims'] = xlims
    spectrum_plot_kw['gamma'] = gamma
    annotation_kw['gamma'] = gamma

    envelope_type = find_value_of(
        property='envelope',
        args=args,
        config=config,
        property_type=str,
        allowed_values=SUPPORTED_ENVELOPES,
        default=None,
    )

    spectrum_plot_kw['envelope_type'] = envelope_type
    annotation_kw['envelope_type'] = envelope_type

    spectrum_plot_kw['sticks_off'] = find_value_of(
        property='sticks_off',
        args=args,
        config=config,
        property_type=bool,
        default=False,
    )

    spectrum_plot_kw['scatter'] = find_value_of(
        property='scatter',
        args=args,
        config=config,
        property_type=bool,
        default=False,
    )

    spectrum_plot_kw['y_offset'] = find_value_of(
        property='y_offset',
        args=args,
        config=config,
        property_type=float,
        default=0.0,
    )

    top_feature = plot_spectra(ax, spectra, **spectrum_plot_kw)

    the_spectrum_assignments = add_spectrum_assignments(
        ax, spectra, top_feature, xlims, spectrum_plot_kw['y_offset'],
    )

    annotation_kw['annotation'] = find_value_of(
        property='annotate',
        config=config,
        args=args,
        property_type=str,
        default="",
    )
    annotation_kw['verbose'] = find_value_of(
        property='verbose',
        config=config,
        args=args,
        property_type=bool,
        default=False,
    )
    annotation_kw['position'] = find_value_of(
        property='position_annotation',
        config=config,
        args=args,
        property_type=str,
        default='top left',
        allowed_values=ALLOWED_ANNOTATION_POSITIONS,
    )
    add_info_text(ax, **annotation_kw,)

    # config file can specify extra "reference" spectra
    texts_ref = list()
    if "reference_spectrum" in config:
        for ref_spec_conf in config['reference_spectrum']:
            spectrum_files = find_value_of(
                property='spectrum_files',
                config=ref_spec_conf,
                property_type=list,
                default=[],
            )
            if len(spectrum_files) == 0:
                print(
                    "Error: No spectrum_files in the 'reference_spectrum'",
                    file=sys.stderr,
                )
                continue
            energy_units = find_energy_units(ref_spec_conf)
            ref_spectra = get_ref_spectrum(spectrum_files, energy_units)

            ref_spectrum_tweaks: SpectrumTweaks = collect_spectrum_tweaks(
                args=None,
                config=ref_spec_conf
            )

            input_shift_eV = find_shift_eV(config, args)
            match_origin = find_match_origin(config, args)
            shift_eV = calculate_spectrum_shift(
                ref_spectra,
                input_shift_eV,
                match_origin
            )
            ref_spectrum_tweaks.set_shift_eV(shift_eV)
            annotation_kw['shift_eV'] = shift_eV

            for spectrum in ref_spectra:
                ref_spectrum_tweaks.apply_to(spectrum)

            ref_spec_plot_kw = spectrum_plot_kw.copy()
            ref_spec_plot_kw['y_offset'] = find_value_of(
                property='y_offset',
                config=ref_spec_conf,
                property_type=float,
                default=0.0,
            )
            ref_spec_plot_kw['sticks_off'] = find_value_of(
                property='sticks_off',
                config=ref_spec_conf,
                property_type=bool,
                default=False,
            )
            ref_spec_plot_kw['envelope_type'] = find_value_of(
                property='envelope',
                config=ref_spec_conf,
                property_type=str,
                allowed_values=SUPPORTED_ENVELOPES,
                default=None,
            )

            ref_spec_plot_kw['line_kwargs'] = find_value_of(
                property='line_kwargs',
                config=ref_spec_conf,
                property_type=dict,
                default={},
            )

            top_feature = plot_spectra(ax, ref_spectra, **ref_spec_plot_kw)

            ref_spectrum_assignments = add_spectrum_assignments(
                ax, ref_spectra, top_feature, xlims,
                ref_spec_plot_kw['y_offset'],
            )
            texts_ref += ref_spectrum_assignments

    if ax2nd is not None:
        print("Ready to print second panel spectrum")
        spectrum_files = find_value_of(
            'second_panel_spectrum_files',
            config,
            args,
            default=[],
        )
        spectrum_format = find_value_of(
            'second_panel_spectrum_format',
            config,
            args,
            property_type=str,
            default='fort.20',
            allowed_values=SUPPORTED_FILETYPES,
        )
        energy_units = find_value_of(
            'second_panel_energy_units',
            config,
            args,
            property_type=str,
            default='eV',
            allowed_values=supported_units,
        )
        second_panel_spectra, basis, lanczos = collect_spectra(
            spectrum_files=spectrum_files,
            spectrum_format=spectrum_format,
            energy_units=energy_units,
        )

        second_panel_input_shift_eV = find_value_of(
            property='second_panel_shift_eV',
            args=args,
            config=config,
            default=input_shift_eV,
            property_type=float,
        )

        second_panel_match_origin = find_value_of(
            property='second_panel_match_origin',
            config=config,
            args=args,
            default=match_origin,
            property_type=float | None,
        )

        second_panel_shift_eV = calculate_spectrum_shift(
            spectra=second_panel_spectra,
            shift_eV=second_panel_input_shift_eV,
            match_origin=second_panel_match_origin,
        )
        spectrum_tweaks.set_shift_eV(second_panel_shift_eV)

        scd_pnl_rescale_intensities = find_value_of(
            property='second_panel_rescale_intensities',
            config=config,
            args=args,
            property_type=float,
            default=None
        )
        spectrum_tweaks.set_rescale_intensities(scd_pnl_rescale_intensities)

        for spectrum in second_panel_spectra:
            spectrum_tweaks.apply_to(spectrum)

        sec_pnnl_plot_kw = spectrum_plot_kw.copy()
        sec_pnnl_plot_kw['y_offset'] = find_value_of(
            property='y_offset',
            args=args,
            config=config,
            property_type=float,
            default=spectrum_plot_kw['y_offset'],
        )
        sec_pnnl_plot_kw['sticks_off'] = find_value_of(
            property='second_panel_sticks_off',
            args=args,
            config=config,
            property_type=bool,
            default=spectrum_plot_kw['sticks_off'],
        )
        sec_pnnl_plot_kw['scatter'] = find_value_of(
            property='second_panel_scatter',
            args=args,
            config=config,
            property_type=bool,
            default=spectrum_plot_kw['scatter'],
        )
        sec_pnnl_plot_kw['envelope_type'] = find_value_of(
            property='second_panel_envelope',
            args=args,
            config=config,
            property_type=str,
            allowed_values=SUPPORTED_ENVELOPES,
            default=spectrum_plot_kw['envelope_type'],
        )

        second_panel_top_feature = plot_spectra(
            ax2nd,
            second_panel_spectra,
            **sec_pnnl_plot_kw)

        assignments_2nd_spectrum = add_spectrum_assignments(
            ax2nd, second_panel_spectra, second_panel_top_feature, xlims
        )

        ax2nd_annotation_kw = annotation_kw.copy()
        ax2nd_annotation_kw['annotation'] = find_value_of(
            property='second_panel_annotate',
            config=config,
            args=args,
            property_type=str,
            default="",
        )

        # Cut out the repeated axis' lables
        for unwanted in ['lanczos', 'basis', 'gamma', 'envelope_type',
                         'verbose', 'shift_eV']:
            if unwanted in ax2nd_annotation_kw:
                del ax2nd_annotation_kw[unwanted]

        # Uncomment if you want it
        # ax2nd_annotation_kw['shift_eV'] = second_panel_shift_eV
        add_info_text(ax2nd, **ax2nd_annotation_kw,)

    spectrum_ax_kw = collect_ax_tweaks(args, config)
    apply_ax_tweaks(ax=ax, xlims=xlims, **spectrum_ax_kw)
    if ax2nd is not None:
        ax.xaxis.set_ticklabels([])
        apply_ax_tweaks(ax=ax2nd, xlims=xlims, **spectrum_ax_kw)

    # Only after xlims were adjusted
    origin_eV = get_origin_eV(spectra)

    second_axis = find_value_of(
        'second_axis',
        config=config,
        args=args,
        property_type=str,
        allowed_values=ALLOWED_SECOND_AXIS_STYLES,
    )

    interval = find_value_of(
        'horizontal_minor_ticks_2nd_axis',
        config,
        args,
        property_type=float | int,
        default=None,
    )
    top_ax = add_second_axis(
        ax=ax,
        origin_eV=origin_eV,
        second_axis=second_axis,
        interval=interval
    )
    if ax2nd is not None:
        ax2nd_top_ax = add_second_axis(
            ax=ax2nd,
            origin_eV=origin_eV,
            second_axis=second_axis,
            interval=interval,
        )
        ax.xaxis.set_ticklabels([])
        ax2nd_top_ax.xaxis.set_ticklabels([])

        for loc_ax in [ax, top_ax, ax2nd, ax2nd_top_ax]:
            loc_ax.tick_params(axis='both', which='both', direction='in')

    # TODO: figure out the rescale_factor
    if len(texts_ref) != 0:
        decongest_assignments(ax, texts_ref)
    if ax2nd is not None:
        decongest_assignments(ax2nd, assignments_2nd_spectrum)

    decongest_assignments(ax, the_spectrum_assignments)

    filename = find_value_of(
        property='filename',
        args=args,
        config=config,
        default=None,
    )
    if filename is None:
        spectrum_files = find_value_of(
            property='spectrum_files',
            args=args,
            config=config,
            default=None,
        )
        filename = prepare_filename(spectrum_files)

    dont_save = find_value_of(
        property='dont_save',
        config=config,
        args=args,
        default=None,
    )

    if dont_save is True:
        plt.show()
    else:
        print(f"File saved as: {filename}")
        plt.savefig(filename)

    return 0


if __name__ == '__main__':
    main()
