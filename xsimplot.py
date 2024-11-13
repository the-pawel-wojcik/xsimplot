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


DISREGARD_INTENSITY = 1e-20
ANNOTATION_DISREGARD_THRESH = 0.0001

COLORS = [color for color in mcolors.TABLEAU_COLORS.keys()]
FONTSIZE = 12
CM2INCH = 1/2.54

CM2eV = 1./8065.543937
eV2CM = 8065.543937

supported_units = {
    "eV": lambda x: x,
    "cm-1": lambda x: x * CM2eV,
    "nm": lambda x: 1239.84198 / x,
}


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
        self.update_intensities_sign_flag()
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
        self.spectral_points.sort(key=lambda x: x['energy'])
        return self.spectral_points[0]['energy']

    def get_max_intensity(self):
        """Returns max of abs intensity."""
        max_peak = max(self.spectral_points, key=lambda x: abs(x['intensity']))
        return max_peak['intensity']

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

    parser.add_argument("output_files",
                        help="List of files with the spectrum."
                        "The -n (--spectrum_format) flag controls the "
                        "spectrum format.",
                        nargs="+")

    parser.add_argument("-a", "--annotate",
                        help="Place string with the annotation on the figure."
                        "Use 'a' as the first letter of the string to append"
                        " the annotation generated by other flags; use 'o' to"
                        " overwrite.",
                        default="",
                        type=str)

    parser.add_argument("-p", "--position_annotation",
                        help="Place the annotation at",
                        choices=[
                            "top left", "top center", "top right",
                            "bottom left", "bottom center", "bottom right"
                        ],
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
                        choices=['stack', 'overlay', 'sum only'])

    parser.add_argument("-g", "--gamma",
                        help="Gamma in the Lorenzian:\n" +
                        "(0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma) **2)",
                        type=float,
                        default=None)

    parser.add_argument("-i", "--all_intensities_even",
                        help="Use this switch to set all intenties to the"
                        " same value.",
                        type=float,
                        default=None)

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

    parser.add_argument("-n", "--spectrum_format",
                        help="Chose the format of the spectrum file. 'xsim'"
                        "requires additional parser. 'ref' marks that the"
                        " input follows the 'reference_spectrum' format"
                        " described in the reamde file. If 'ref' uses peak "
                        " location not in 'eV', see '-u'.",
                        default=None,  # defaults to fort.20 see code
                        type=str,
                        choices=["xsim", "fort.20", "ezFCF", 'ref'])

    parser.add_argument("-r", "--scale_factor",
                        help="Scale the figure size with the factor.",
                        default=None,
                        type=float)

    parser.add_argument("-t", "--sticks_off",
                        help="Do NOT show stick spectrum.",
                        default=False,
                        action="store_true")

    parser.add_argument("-u", "--energy_units",
                        help="For use with '-n ref';"
                        " allows to change energy units of the spectrum"
                        " (defaults to eV).",
                        default=None,  # defaults to eV
                        choices=supported_units,
                        )

    parser.add_argument("-v", "--verbose",
                        help="Annotate with # of Lanczos it and basis size.",
                        default=False,
                        action="store_true")

    save = parser.add_mutually_exclusive_group()

    save.add_argument("-d", "--dont_save",
                      help="Just show the figure. Don't save it yet.",
                      default=False,
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
                        choices=["cm", "cm offset", "nm"],
                        default=None)

    parser.add_argument("-y", "--show_yaxis_ticks",
                        help="Show ticks on the yaxis.",
                        default=None,
                        type=bool)

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
        color: str = 'k',
        **line_kwargs
):
    for spectral_point in peaks:
        energy_eV = spectral_point['energy']
        intensity = spectral_point['intensity']
        ax.vlines(x=energy_eV, ymin=0.0, ymax=intensity, colors=color)


def get_xsim_outputs_from_fort20(
        args: argparse.Namespace,
) -> tuple[list[Spectrum], int]:
    xsim_outputs = []
    lanczos = None
    for file_idx, fort20_file in enumerate(args.output_files):

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
        args: argparse.Namespace,
) -> tuple[list[Spectrum], str, int]:

    from cfour_parser.xsim import parse_xsim_output
    xsim_outputs = []
    basis = None
    lanczos = None
    for file_idx, out_file in enumerate(args.output_files):
        with open(out_file) as f:
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


def get_ref_spectrum(
        filename: str,
        convert_to_eV
) -> Spectrum:
    ref_spectrum = Spectrum()
    assignments_are_available = False
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


def get_ref_spectrum_from_args_and_config(
        args: argparse.Namespace,
        config: dict,
) -> list[Spectrum]:
    """
    spectrum as csv
    energy,intensity,assignmnet
    assignment is optional
    """
    energy_units = None
    if 'energy_units' in config:
        if config['energy_units'] not in supported_units:
            print("Error: the config file requests unsupported energy units:\n"
                  f"\tenergy_units = {config['energy_units']}",
                  file=sys.stderr)
            sys.exit(1)
        energy_units = config['energy_units']
    if args.energy_units is not None:
        energy_units = args.energy_units
    if energy_units is None:
        energy_units = 'eV'

    convert_to_eV = supported_units[energy_units]

    spectra = []
    for file_idx, out_file in enumerate(args.output_files):
        ref_spectrum = get_ref_spectrum(out_file, convert_to_eV)
        spectra += [ref_spectrum]

    return spectra


def get_ezFCF_spectrum(
        args: argparse.Namespace
) -> list[Spectrum]:
    spectra = []
    for file_idx, out_file in enumerate(args.output_files):
        with open(out_file) as f:
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


def find_shift(
        spectra: list[Spectrum],
        args: argparse.Namespace,
        config: dict,
) -> float:
    """
    Find shift (in eV) which will be applied to the spectrum.
    """

    # Command line args are the most important
    if args.shift_eV is not None:
        return args.shift_eV

    first_peak_position = None
    if 'match_origin' in config:
        first_peak_position = config['match_origin']
    if args.match_origin is not None:
        first_peak_position = args.match_origin

    if first_peak_position is not None:
        origins = []
        for spectrum in spectra:
            min_element = min(
                spectrum.spectral_points,
                key=lambda x: x['energy']
            )
            origins.append(min_element['energy'])
        origin_eV = min(origins)
        shift_eV = origin_eV - first_peak_position
        return shift_eV

    return None


def find_left_right_gamma(
        spectra: list[Spectrum],
        args: argparse.Namespace,
        config: dict,
        how_far: int = 4,
) -> tuple[float, float, float]:
    """
    Find the positions of the lowest energy and highest energy peaks and return
    values how_far*gamma away from them as the plot limits.
    """

    gamma_eV = find_gamma(args, config)

    if 'xlims' in config:
        left = config['xlims']['left']
        right = config['xlims']['right']
        return (left, right, gamma_eV)

    # Default values
    mins = list()
    maxes = list()

    for spectrum in spectra:
        min_element = min(spectrum.spectral_points, key=lambda x: x['energy'])
        max_element = max(spectrum.spectral_points, key=lambda x: x['energy'])
        mins.append(min_element['energy'])
        maxes.append(max_element['energy'])

    minimum = min(mins)
    maximum = max(maxes)
    left = minimum - how_far * gamma_eV
    right = maximum + how_far * gamma_eV

    return (left, right, gamma_eV)


def apply_shift(
        spectra: list[Spectrum],
        shift_eV: float,
) -> list:
    if shift_eV is None:
        return spectra

    for spectrum in spectra:
        for peak in spectrum.spectral_points:
            peak['energy'] -= shift_eV

    return spectra


def add_envelope(
        ax: mpl.axes.Axes,
        args: argparse.Namespace,
        config: dict,
        spectra: list[Spectrum],
        xlims: list[float],
        gamma: float,
) -> float:
    """ Returns max value of the envelope. """

    envelope_type = None
    if 'envelope' in config:
        envelope_type = config['envelope']
    # Command line args override config options
    if args.envelope is not None:
        envelope_type = args.envelope

    if envelope_type is None:
        return 0.0

    npoints = 1500
    xs = np.linspace(xlims[0], xlims[1], npoints)
    accumutated_ys = np.zeros_like(xs)

    for spectrum_idx, spectrum in enumerate(spectra):
        # TODO: remove COLORS here
        if spectrum_idx == len(COLORS) or spectrum_idx > len(COLORS):
            print("Too many colors already.", file=sys.stderr)
            sys.exit(1)

        state_spectrum = [lorenz_intensity(x, gamma, spectrum) for x in xs]
        state_spectrum = np.array(state_spectrum)
        color = COLORS[spectrum_idx]
        if envelope_type == "stack":
            ax.fill_between(xs, accumutated_ys + state_spectrum,
                            accumutated_ys, color=color, alpha=0.2)
        elif envelope_type == "overlay":
            ax.fill_between(xs, state_spectrum, np.zeros_like(xs), color=color,
                            alpha=0.2)
        accumutated_ys += state_spectrum

    # dashed = (0, (5, 5))
    # densly_dashdotted = (0, (3, 1, 1, 1))
    # Plot the total spectrum extra for overlay
    if envelope_type in ["overlay", "sum only"]:
        ax.plot(xs, accumutated_ys, color='tab:gray', lw=1)

    fig_max_y = np.max(accumutated_ys)

    return fig_max_y


def add_peaks(
        ax: mpl.axes.Axes,
        args: argparse.Namespace,
        config: dict,
        spectra: list[Spectrum],
        xlims: list[float],
) -> float:
    """ Returns the height of the tallest added peak. """
    sticks_off = False
    if 'sticks_off' in config:
        sticks_off = config['sticks_off']
    if args.sticks_off is True:
        sticks_off = True

    if sticks_off is True:
        return 0.0

    peaks_maxima = []
    for spectrum_idx, spectrum in enumerate(spectra):
        # TODO: remove COLORS in here
        if spectrum_idx == len(COLORS) or spectrum_idx > len(COLORS):
            print("Too many colors already.", file=sys.stder)
            sys.exit(1)

        my_peaks = [
            peak for peak in spectrum.spectral_points if
            peak['energy'] >= xlims[0] and peak['energy'] <= xlims[1]
        ]
        line_kwargs = {
            'color':  COLORS[spectrum_idx],
        }
        stem_spectral_peaks(my_peaks, ax, **line_kwargs)

        if len(my_peaks) == 0:
            max_peak = {'intensity': 0.0}
        else:
            max_peak = max(my_peaks, key=lambda x: x['intensity'])

        peaks_maxima.append(max_peak['intensity'])

    return max(peaks_maxima)


def add_info_text(
        ax: mpl.axes.Axes,
        args: argparse.Namespace,
        config: dict,
        shift_eV: float,
        basis: str,
        lanczos: int,
        gamma: float,
):

    info_kwargs = {
        'fontsize': FONTSIZE,
        'color': 'k',
        'transform': ax.transAxes,
    }

    x_position = {
        "top left": 0.01,
        "top center": 0.5,
        "top right": 0.99,
        "bottom left": 0.01,
        "bottom center": 0.5,
        "bottom right": 0.99,
    }

    y_position = {
        "top left": 0.99,
        "top center": 0.99,
        "top right": 0.99,
        "bottom left": 0.01,
        "bottom center": 0.01,
        "bottom right": 0.01,
    }

    position = "top left"  # default
    if 'position_annotation' in config:  # allow to set it from the config
        position = config['position_annotation']
        if position not in x_position:
            print("Error: Invalid option present in the config file:\n\t"
                  f"position_annotation = '{position}'\n\t"
                  f"Allowed values: {", ".join(x_position.keys())}",
                  file=sys.stderr)
            sys.exit(1)
    if args.position_annotation is not None:  # command line can overwrite
        position = args.position_annotation

    info_kwargs['horizontalalignment'] = position.split()[1]
    info_kwargs['verticalalignment'] = position.split()[0]

    text = ""

    envelope_type = None
    if 'envelope' in config:
        envelope_type = config['envelope']
    if args.envelope is not None:
        envelope_type = args.envelope

    if envelope_type is not None:
        text = r'$\gamma = ' + f'{gamma:.3f}$\n'

    if shift_eV is not None:
        text += f'$s = {shift_eV:.2f}$ eV'

    verbose = False
    if 'verbose' in config:
        verbose = config['verbose']
    if args.verbose is True:
        verbose = True

    if verbose is True:
        if basis is not None:
            text += f'\nBasis: {basis.split()[0]}'
        if lanczos is not None:
            text += f'\nLanczos: {lanczos}'

    annotation = ""
    if 'annotate' in config:
        annotation = config['annotate']
    if args.annotate != "":
        annotation = args.annotate

    if annotation != "":
        text_with_newline = "\n".join(annotation[1:].split(r'\n'))
        # append
        if annotation[0] == "a":
            text += "\n" + text_with_newline
        # overwrite
        elif annotation[0] == "o":
            text = text_with_newline
        else:
            print("Warning: The annotation texts needs to start with either"
                  " 'a' or 'o', see help for details.", file=sys.stderr)

    ax.text(x_position[position], y_position[position], text, **info_kwargs)


def prepare_filename(
        args: argparse.Namespace,
        config: dict,
) -> str:
    user_filename = None
    if 'filename' in config:
        user_filename = config['filename']
    if args.filename is not None:
        user_filename = args.filename

    if user_filename is not None:
        return user_filename

    # If the user does not say the name create one yourself
    path = os.path.expanduser('~')

    filename = path + "/"
    for idx, outname in enumerate(args.output_files):
        if idx > 0:
            filename += "+"
        filename += os.path.basename(outname)
    filename += '_spectrum.pdf'

    return filename


def find_gamma(args: argparse.Namespace, config: dict) -> float:
    gamma = None
    if 'gamma' in config:
        gamma = config['gamma']

    # Command line argument takes precedence
    if args.gamma is not None:
        gamma = args.gamma

    # default value
    if gamma is None:
        gamma = 0.03

    return gamma


def get_origin(spectra: list[Spectrum]) -> float:
    origin = None
    for spectrum in spectra:
        loc_origin = spectrum.get_origin_energy()
        if origin is None:
            origin = loc_origin
        elif origin > loc_origin:
            origin = loc_origin

    return origin


def add_minor_ticks(
        args: argparse.Namespace,
        config: dict,
        ax: mpl.axes.Axes,
):
    interval = None
    if 'horizontal_minor_ticks' in config:
        interval = config['horizontal_minor_ticks']
    if args.horizontal_minor_ticks is not None:
        interval = args.horizontal_minor_ticks

    if interval is None:
        return

    ax.xaxis.set_minor_locator(MultipleLocator(interval))


def add_second_axis(
        args: argparse.Namespace,
        config: dict,
        ax: mpl.axes.Axes,
        origin_eV: float,
) -> mpl.axes.Axes:

    second_axis = None

    if 'second_axis' in config:
        second_axis = config['second_axis']
    if args.second_axis is not None:
        second_axis = args.second_axis

    if second_axis is None:
        return None

    if second_axis == "cm offset":
        second_ax: mpl.axes.Axes = add_cm_scale(ax, args, config, origin_eV)

    elif second_axis == "cm":
        second_ax: mpl.axes.Axes = add_cm_scale(ax, args, config)

    elif second_axis == "nm":
        second_ax: mpl.axes.Axes = add_nm_scale(args, ax)

    return second_ax


def add_cm_scale(
        ax: mpl.axes.Axes,
        args: argparse.Namespace,
        config: dict,
        origin_eV: float = None,
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

    interval = None
    if 'horizontal_minor_ticks_2nd_axis' in config:
        interval = config['horizontal_minor_ticks_2nd_axis']
    if args.horizontal_minor_ticks_2nd_axis is not None:
        interval = args.horizontal_minor_ticks_2nd_axis

    if interval is not None:
        ax_cm.xaxis.set_minor_locator(MultipleLocator(interval))

    return ax_cm


def add_nm_scale(
        args: argparse.Namespace,
        ax: mpl.axes.Axes,
):
    r""" Relation between photon's energy and wavelength:
        E = hc / \lambda
    """
    ToLambda = 1239.84198  # from eV to nm
    ax_nm = ax.twiny()

    x1, x2 = ax.get_xlim()
    x1 = ToLambda / x1
    x2 = ToLambda / x2

    # TODO: I don't think this works. I think that you leave the scale linear
    # in energy.
    ax_nm.set_xlim(x1, x2)


def get_fig_and_ax(
        args: argparse.Namespace,
        config: dict,
) -> tuple[mpl.figure.Figure, mpl.axis.Axis]:

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
    FIGSIZE = 8 * CM2INCH * scale_factor
    fig, ax = plt.subplots(figsize=(FIGSIZE * aspect_ratio, FIGSIZE),
                           layout='constrained')
    return fig, ax


def collect_reference_peaks_from_config(
        args: argparse.Namespace,
        config: dict,
) -> Spectrum:
    if "reference_peaks" not in config:
        return None

    spectrum = Spectrum()
    # TODO: assure that reference_peaks are propertly formatted.
    peaks = config['reference_peaks']
    for peak in peaks:
        energy = peak['energy']
        energy_unit = peak['energy_unit']
        amplitude = peak['amplitude']
        assignment = peak['assignment']

        if energy_unit not in supported_units:
            print(f"Error: energy units other than {supported_units.keys()}"
                  " are not supported in 'reference_peaks' array.",
                  file=sys.stderr)
            sys.exit(1)
        x_eV = supported_units[energy_unit](energy)

        spectrum.add_sepctral_point(
            energy=x_eV,
            intensity=amplitude,
            assignment=assignment,
        )

    return spectrum


def collect_ref_spectrum_config(
        spectrum: dict,
) -> dict:
    """
    Returns
    ```python
    ref_spectrum_config = {
        'unit': str,  # checked
        'rescale_factor': float,
        'plot_type': str,  # checked
        'y_offset': float,
        'file_name': str,
        'match_origin': None or float,
        'line_kwargs': dict,
    }
    ```
    """
    unit = 'eV'
    if 'energy_units' in spectrum:
        unit = spectrum['energy_units']
        if unit not in supported_units:
            print("Error: energy units other than "
                  f"{supported_units.keys()}"
                  " are not supported by 'reference_spectrum'.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("Info: Energy unit in the reference_spectrum section of the"
              " config file is not specified. Using the default: 'eV'.",
              file=sys.stderr)

    rescale_factor = 1.0
    if 'rescale_intensities' in spectrum:
        rescale_factor = spectrum['rescale_intensities']
    else:
        print("Info: rescale_intensities not specified in the "
              "reference_spectrum section of the config file."
              "Using intensities from the csv file.",
              file=sys.stderr)

    match_origin = None
    if 'match_origin' in spectrum:
        match_origin = float(spectrum['match_origin'])
        print("Info: Matching the reference spectrum against the origin at"
              f" {match_origin} {unit}.",
              file=sys.stderr)

    line_kwargs = {}
    if 'line_kwargs' in spectrum:
        for key, value in spectrum['line_kwargs'].items():
            line_kwargs[key] = value

    plot_type = 'stems'
    if 'plot_type' in spectrum:
        plot_type = spectrum['plot_type']
    else:
        print("Info: plot_type not specified in the "
              "reference_spectrum section of the config file."
              "Using the default.",
              file=sys.stderr)

    if plot_type not in ['scatter', 'stems', 'plot']:
        print("Error: The only supported values of 'plot_type' in the "
              "'reference_spectrum' part of the config file are 'scatter'"
              ", 'stems', and 'plot' ", file=sys.stderr)
        sys.exit(1)

    if 'file' not in spectrum:
        print("Error: 'reference_spectrum' is missing the 'file' line.",
              file=sys.stderr)
        sys.exit(1)

    y_offset = 0.0
    if 'y_offset' in spectrum:
        y_offset = spectrum['y_offset']

    file_name = spectrum['file']
    file_name = os.path.expanduser(file_name)

    ref_spectrum_config = {
        'unit': unit,
        'rescale_factor': rescale_factor,
        'plot_type': plot_type,
        'y_offset': y_offset,
        'file_name': file_name,
        'match_origin': match_origin,
        'line_kwargs': line_kwargs,
    }

    return ref_spectrum_config


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
            text_kwargs['va'] = 'top'
        else:
            text_kwargs['va'] = 'bottom'

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


def set_limits(args: argparse.Namespace, ax, xlims):
    ax.set_xlim([xlims[0], xlims[1]])


def get_config(args: argparse.Namespace) -> dict:
    config_file = os.path.expanduser(args.config)
    if os.path.isfile(config_file) is False:
        print(f"Info: No config file {args.config} present.", file=sys.stderr)
        return {}

    print(f"Info: Using the config file {args.config}.", file=sys.stderr)
    with open(config_file, 'rb') as config_toml:
        config = tomllib.load(config_toml)

    return config


def customize_yaxis(
        args: argparse.Namespace,
        config: dict,
        ax: mpl.axes.Axes,
):
    show_yaxis_ticks = None
    if 'show_yaxis_ticks' in config:
        show_yaxis_ticks = config['show_yaxis_ticks']
    if args.show_yaxis_ticks is not None:
        show_yaxis_ticks = args.show_yaxis_ticks

    if show_yaxis_ticks is None:
        show_yaxis_ticks = False  # set the default

    if show_yaxis_ticks is False:
        ax.get_yaxis().set_ticks([])

    if 'ylims' in config:
        ylims = config['ylims']
        ax.set_ylim(bottom=ylims['bottom'], top=ylims['top'])


def collect_spectra(
        args: argparse.Namespace,
        config: dict,
) -> tuple[list[Spectrum], str, int]:
    """Collect the spectrum specified on the command line."""

    spectrum_format = None
    if 'spectrum_format' in config:
        if config['spectrum_format'] not in [
                'xsim', 'fort.20', 'ezFCF', 'ref'
        ]:
            print("Error: the config file contains invalid spectrum format\n"
                  f"\tspectrum_format = {config['spectrum_format']}",
                  file=sys.stderr)
            sys.exit(1)
        spectrum_format = config['spectrum_format']
    # The command line can override it
    if args.spectrum_format is not None:
        spectrum_format = args.spectrum_format
    # if not specified it defaults to fort.20
    if spectrum_format is None:
        spectrum_format = "fort.20"

    if spectrum_format == "fort.20":
        basis = None
        spectra, lanczos = get_xsim_outputs_from_fort20(args)
    elif spectrum_format == "xsim":
        spectra, basis, lanczos = get_xsim_outputs(args)
    elif spectrum_format == "ezFCF":
        basis = None
        lanczos = None
        spectra = get_ezFCF_spectrum(args)
    elif spectrum_format == "ref":
        basis = None
        lanczos = None
        spectra = get_ref_spectrum_from_args_and_config(args, config)
    else:
        print(f"Error: Unknown spectrum format {spectrum_format}",
              file=sys.stderr)

    return spectra, basis, lanczos


def set_intensities(
        args: argparse.Namespace,
        config: dict,
        spectra: list[Spectrum],
):
    """
    Sometimes only the position of peaks is meaningful. Check if the switch
    'all_intensities_even' is set to any value is apply it.
    """
    even_intensites = None
    if 'all_intensities_even' in config:
        even_intensites = config['all_intensities_even']

    if args.all_intensities_even is not None:
        even_intensites = args.all_intensities_even

    if even_intensites is None:
        return spectra

    for spectrum in spectra:
        for peak in spectrum.spectral_points:
            peak['intensity'] = even_intensites

    return spectra


def rescale_intensities(
        args: argparse.Namespace,
        config: dict,
        spectra: list[Spectrum],
) -> list:
    intensites_factor = None
    if 'rescale_intensities' in config:
        intensites_factor = config['rescale_intensities']

    if args.rescale_intensities is not None:
        intensites_factor = args.rescale_intensities

    if intensites_factor is None:
        return spectra

    for spectrum in spectra:
        for peak in spectrum.spectral_points:
            peak['intensity'] *= intensites_factor

    return spectra


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

    adjust_text(texts, ax=ax,
                # objects=objects,
                avoid_self=True,
                only_move=only_move,
                min_arrow_len=20,
                arrowprops=dict(arrowstyle='->',
                                color='lightgray',
                                lw=1,
                                mutation_scale=5,
                                linestyle='-'
                                )
                )


def main():
    args = parse_command_line()
    config = get_config(args)

    spectra, basis, lanczos = collect_spectra(args, config)

    shift_eV = find_shift(spectra, args, config)
    *xlims, gamma = find_left_right_gamma(spectra, args, config)
    spectra = apply_shift(spectra, shift_eV)
    spectra = set_intensities(args, config, spectra)
    spectra = rescale_intensities(args, config, spectra)

    fig, ax = get_fig_and_ax(args, config)

    envelope_max_y = add_envelope(ax, args, config, spectra, xlims, gamma)
    max_peak = add_peaks(ax, args, config, spectra, xlims)
    top_feature = max(envelope_max_y, max_peak)
    spectrum_texts = add_spectrum_assignments(ax, spectra, top_feature, xlims)

    add_info_text(ax, args, config, shift_eV, basis, lanczos, gamma)

    # config file can specify extra peaks
    reference_peaks = collect_reference_peaks_from_config(args, config)
    if reference_peaks is not None:
        line_kwargs = {
            'color': 'k',
            'linestyles': 'solid',
        }
        stem_spectral_peaks(reference_peaks.spectral_points, ax, **line_kwargs)
        add_spectrum_assignments(ax, [reference_peaks], top_feature, xlims)

    # config file can specify extra "reference" spectra
    texts_ref = list()
    rescale_factor = None
    if "reference_spectra" in config:
        for ref_spec_config_toml in config['reference_spectra']:
            ref_spec_conf = collect_ref_spectrum_config(ref_spec_config_toml)

            unit = ref_spec_conf['unit']
            energy_to_eV = supported_units[unit]
            file_name = ref_spec_conf['file_name']
            ref_spec: Spectrum = get_ref_spectrum(file_name, energy_to_eV)

            shift = 0.0
            if ref_spec_conf['match_origin'] is not None:
                match_origin_eV = energy_to_eV(ref_spec_conf['match_origin'])
                shift = ref_spec.get_origin_energy() - match_origin_eV
            ref_spec -= shift

            rescale_factor = ref_spec_conf['rescale_factor']
            ref_spec *= rescale_factor

            spectrum_list = [
                [row['energy'], row['intensity'],]
                for row in ref_spec.spectral_points
            ]
            xs, ys = [list(a) for a in zip(*spectrum_list)]

            line_kwargs = {}
            line_kwargs.update(ref_spec_conf['line_kwargs'])

            # TODO: make it a single function call and use it with the
            # previous spectra

            plot_type = ref_spec_conf['plot_type']
            y_offset = ref_spec_conf['y_offset']

            if plot_type == 'stems':
                peak_lines: mpl.collections.LineCollection = ax.vlines(
                    xs,
                    [y_offset for _ in ys],
                    [y + y_offset for y in ys],
                    label=None,
                    **line_kwargs
                )

            elif plot_type == "scatter":
                ax.scatter(xs, [y_offset + y for y in ys])

            elif plot_type == "plot":
                ax.plot(xs, [y_offset + y for y in ys])

            ref_spec.update_assignments_flag()
            texts_ref += add_spectrum_assignments(
                ax,
                [ref_spec],
                top_feature=0.0,
                xlims=xlims,
                y_offset=y_offset,
            )

    set_limits(args, ax, xlims)

    origin = get_origin(spectra)
    add_minor_ticks(args, config, ax)
    add_second_axis(args, config, ax, origin)

    customize_yaxis(args, config, ax)

    # TODO: figure out the rescale_factor
    if rescale_factor is not None:
        decongest_assignments(ax, texts_ref, rescale_factor > 0)
    decongest_assignments(ax, spectrum_texts)

    filename = prepare_filename(args, config)
    dont_save = False
    if 'dont_save' in config:
        dont_save = config['dont_save']
    if args.dont_save is True:
        dont_save = True

    if dont_save is True:
        plt.show()
    else:
        print(f"File saved as: {filename}")
        plt.savefig(filename)

    return 0


if __name__ == '__main__':
    main()
