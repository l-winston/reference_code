#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PulsON 440 radar data unpacking."""

__author__ = "Ramamurthy Bhagavatula, Michael Riedl"
__version__ = "1.0"
__maintainer__ = "Ramamurthy Bhagavatula"
__email__ = "ramamurthy.bhagavatula@ll.mit.edu"

# Update path
from pathlib import Path, PurePosixPath
import sys
if Path('..//').resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path('..//').resolve().as_posix())

# Import required modules and methods
import argparse
from common.constants import SPEED_OF_LIGHT
from common.helper_functions import is_valid_file, deconflict_file
from math import floor
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
import numpy as np
import pickle
from pulson440.formats import MRM_GET_CONFIG_CONFIRM
from pulson440.constants import T_BIN, DT_0, MAX_SCAN_INFO_PACKET_SIZE, DEFAULT_CONFIG, BYTE_ORDER
from scipy.signal import hilbert

# Defaults
DEFAULT_NUM_CPI_PULSES = 32

class ValueFormatter(Formatter):
    """Tick label formatter class specific to formatting range/range bin and time/pulse number ticks
    for displaying RTIs.
    """

    def __init__(self, values):
        """value_formatter class initialization method.

        Args:
            values (list)
                Values to reference in formatting ticks.
        """
        self.values = values

    def __call__(self, x, pos=0):
        """Return formatted tick label.

        Args:
            x (numeric)
                Tick value to format.

            pos (int)
                Tick position.

        Returns:
            tick_label (str)
                Formatted tick label.
        """
        ind = int(np.round(x))
        if ind >= len(self.values) or ind < 0:
            return ''
        return '{0:.1f}  [{1:d}]'.format(self.values[ind], ind)

def read_config_data(file_handle):
    """Read in configuration data based on platform.

    Args:
        file_handle (file object)
            File handle to data to unpack.

    Returns:
        config (dict)
            Radar configuration read from the data.
    """
    # Unpack configuration
    config = dict.fromkeys(DEFAULT_CONFIG)
    for config_field in DEFAULT_CONFIG:
        dtype = MRM_GET_CONFIG_CONFIRM['packet_def'][config_field]
        config_field_bytes = file_handle.read(dtype.itemsize)
        config[config_field] = int.from_bytes(config_field_bytes, byteorder=BYTE_ORDER,
                signed=np.issubdtype(dtype, np.signedinteger))
    return config

def unpack(scan_data_filename):
    """Unpacks PulsOn 440 radar data from input data file.

    Args:
        scan_data_filename (str)
            Path and name to scan data file.

    Returns:
        data (dict)
            Data unpacked from data file.
    """
    with open(scan_data_filename, 'rb') as f:

        # Read configuration part of data
        config = read_config_data(f)

        # Compute range bins in datas
        scan_start_time = float(config['scan_start'])
        start_range = SPEED_OF_LIGHT * ((scan_start_time * 1e-12) - DT_0 * 1e-9) / 2

        # Initialize container for unpacked data
        data = dict()
        data = {'scan_data': [],
                'timestamps': [],
                'pulse_idx': None,
                'range_bins': None,
                'packet_idx': [],
                'config': config}
        single_scan_data = []
        packet_count = 0
        pulse_count = 0

        # Read data
        while True:

            # Read a single data packet and break loop if not a complete packet (in terms of size)
            packet = f.read(MAX_SCAN_INFO_PACKET_SIZE)

            if len(packet) < MAX_SCAN_INFO_PACKET_SIZE:
                break

            # Get information from first packet about how scans are stored and range bins collected
            if packet_count == 0:
                num_range_bins = np.frombuffer(packet[44:48], dtype='>u4')[0]
                num_packets_per_scan = np.frombuffer(packet[50:52], dtype='>u2')[0]
                drange_bins = SPEED_OF_LIGHT * T_BIN * 1e-9 / 2
                range_bins = (start_range + drange_bins * np.arange(0, num_range_bins, 1))

            # Number of samples in current packet, timestamp, and packet index
            num_samples = np.frombuffer(packet[42:44], dtype='>u2')[0]
            timestamp = np.frombuffer(packet[8:12], dtype='>u4')[0]
            data['packet_idx'].append(np.frombuffer(packet[48:50], dtype='>u2')[0])

            # Extract radar data samples from current packet; process last packet within a scan
            # seperately to get all data
            packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], dtype='>i4')
            single_scan_data.append(packet_data)
            packet_count += 1

            if packet_count % num_packets_per_scan == 0:
                data['scan_data'].append(np.concatenate(single_scan_data))
                data['timestamps'].append(timestamp)
                single_scan_data = []
                pulse_count += 1

        # Add last partial scan if present
        if single_scan_data:
            single_scan_data = np.concatenate(single_scan_data)
            num_pad = data['scan_data'][0].size - single_scan_data.size
            single_scan_data = np.pad(single_scan_data, (0, num_pad), 'constant', constant_values=0)
            data['scan_data'].append(single_scan_data)
            data['timestamps'].append(timestamp)
            pulse_count += 1

        # Stack scan data into 2-D array (rows -> pulses, columns -> range bins)
        data['scan_data'] = np.stack(data['scan_data'])

        # Finalize entries in data
        data['timestamps'] = np.asarray(data['timestamps'])
        data['pulse_idx'] = np.arange(0, pulse_count)
        data['range_bins'] = range_bins
        data['packet_idx'] = np.asarray(data['packet_idx'])

        return data

def generate_range_doppler(data, num_pulses_per_cpi=DEFAULT_NUM_CPI_PULSES, keep_clutter=False):
    """Generate range-Doppler images from scan data.

    Args:
        data (dict)
            Data unpacked from the scan data file.

        num_pulses_per_cpi (int)
            Number of pulses to use in each CPI. Defaults to 32.

        keep_clutter (bool)
            Indicates whether or not to keep the clutter (zero Doppler) ridge. Defaults to False.

    Returns:
        range_doppler (list)
            Range-Doppler image for each CPI.

        doppler_bins (ndarray)
            Doppler bin values (Hz).

        cpi_timestamps (ndarray)
            Start timestamp of each CPI since first pulse.

    Raises:
        IndexError if number of pulses per CPI exceeds total number of pulses.
    """
    # Check if reqeuested number of pulses per CPI exceed total number of pulses
    if num_pulses_per_cpi > data['pulse_idx'].size:
        raise IndexError('{0} pulses per CPI requested; exceeds {1} total pulses!'.format(
                num_pulses_per_cpi, data['pulse_idx'].size))

    # Determine coarse Doppler bins
    doppler_bins = (np.arange(-num_pulses_per_cpi / 2, num_pulses_per_cpi / 2 - 1) /
                    (np.mean(np.diff(data['timestamps'] / 1000)) * num_pulses_per_cpi))

    # Convert to I/Q using Hilbert transform
    iq_data = hilbert(data['scan_data'], axis=1)

    # Compute range-Doppler for each CPI
    num_cpi = floor(data['pulse_idx'].size / num_pulses_per_cpi)
    cpi_pulse_idx = np.arange(num_pulses_per_cpi)
    range_doppler = [None] * num_cpi
    cpi_timestamps = np.zeros(num_cpi)
    for ii in range(num_cpi):
        start_pulse_idx = ii * num_pulses_per_cpi
        cpi_timestamps[ii] = data['timestamps'][start_pulse_idx]
        range_doppler[ii] = np.fft.fft(iq_data[start_pulse_idx + cpi_pulse_idx, :], axis=0)
        range_doppler[ii] = np.abs(range_doppler[ii])

        # Zero clutter if requested
        if not keep_clutter:
            range_doppler[ii][0, :] = 0

        # Set 0 Hz as center of Doppler axis
        range_doppler[ii] = np.transpose(np.fft.fftshift(range_doppler[ii], axes=0))

    cpi_timestamps -= cpi_timestamps[0]
    return range_doppler, doppler_bins, cpi_timestamps

class RangeDopplerPlot:
    """Class for animating range-Doppler plot."""

    def __init__(self, range_doppler, cpi_timestamps, range_formatter, doppler_formatter):
        """Initialization method."""
        # Raw data
        self.range_doppler = range_doppler
        self.cpi_timestamps = cpi_timestamps

        # Determine min-max valus
        self.min_val = float('inf')
        self.max_val = float('-inf')
        for ii in range(len(range_doppler)):
            self.range_doppler[ii] = 20 * np.log10(range_doppler[ii])
            min_val = np.amin(self.range_doppler[ii][self.range_doppler[ii] != float('-inf')])
            max_val = np.amax(self.range_doppler[ii][self.range_doppler[ii] != float('-inf')])
            if min_val < self.min_val:
                self.min_val = min_val
            if max_val > self.max_val:
                self.max_val = max_val

        # Initialize plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.img = self.ax.imshow(range_doppler[ii], vmin=self.min_val, vmax=self.max_val,
                                 aspect='auto')
        self.ax.set_title('Range-Doppler Images - CPI: #{0}, Times Elapsed: {1} (s)'.format(0,
                          self.cpi_timestamps[0] / 1000))
        self.ax.set_xlabel('Doppler Frequency (Hz) [Doppler Bin Number]')
        self.ax.set_ylabel('Range (m) [Range Bin Number]')
        self.ax.xaxis.set_major_formatter(doppler_formatter)
        self.ax.yaxis.set_major_formatter(range_formatter)
        self.cbar = self.fig.colorbar(self.img)
        self.cbar.ax.set_ylabel('dB')
        self.anim = None

    def _update(self, ii):
        """Update plot method."""
        self.img.set_data(self.range_doppler[ii])
        self.ax.set_title('Range-Doppler Images - CPI: #{0}, Time Elapsed: {1} (s)'.format(ii,
                          self.cpi_timestamps[ii] / 1000))
        plt.draw()

    def start(self):
        """Start range-Doppler plot animation."""
        self.anim = FuncAnimation(self.fig, self._update, len(self.range_doppler),
                                  interval=500, repeat=True)
        plt.draw()
        plt.show()

    def save(self, filename):
        """Save animation to file."""
        self.anim.save(filename, writer=PillowWriter(fps=2))

def parse_args(args):
    """Input argument parser.

    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].

    Returns:
        parsed_args (namespace)
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='PulsON 440 radar data unpacking')
    parser.add_argument('scan_data_filename', help='Path and name of PulsON 440 scan data file')
    parser.add_argument('unpacked_data_filename', nargs='?', default=None,
                        help=('Path and name of file to which data will be saved to; defaults to ' +
                              'not saving it'))
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Plot RTI and range-Doppler of unpacked data; will block computation')
    parser.add_argument('-d', '--delete', action='store_true',
                        help="Don't save the unpacked data")
    parser.add_argument('--num_pulses_per_cpi', type=int, nargs='?', const=DEFAULT_NUM_CPI_PULSES,
                        default=DEFAULT_NUM_CPI_PULSES,
                        help=(('Number of pulses to use per CPI in generating range-Doppler ' +
                               'maps; defaults to {0}').format(DEFAULT_NUM_CPI_PULSES)))
    parser.add_argument('--keep_clutter', action='store_true',
                        help=('Keep clutter ridge (0 Hz Doppler) in range-Doppler images. Only ' +
                              'applies if visualize flag is set; defaults to False'))
    parsed_args = parser.parse_args(args)

    if parsed_args.delete:
        parsed_args.unpacked_data_filename = None
    else:
        if parsed_args.unpacked_data_filename == None:
            parsed_args.unpacked_data_filename = PurePosixPath(parsed_args.scan_data_filename).stem + ".pkl"

    # Check if files are accessible
    is_valid_file(parser, parsed_args.scan_data_filename, 'r')
    is_valid_file(parser, parsed_args.unpacked_data_filename, 'w')

    return parsed_args

def main(args):
    """Main execution method to unpack data, visualize, and save as specified.

    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].

    Returns:
        data (dict)
            Data unpacked from the scan data file.
    """
    # Parse input arguments
    args = parse_args(args)

    # Unpack log file
    data = unpack(args.scan_data_filename)

    # Save (pickle) unpacked data if requested
    if args.unpacked_data_filename:
        with open(args.unpacked_data_filename, 'wb') as f:
            pickle.dump(data, f)

    # Visualize data as RTI and range-Doppler
    plt.ioff()
    if args.visualize:

        # Extract range-Doppler images
        range_doppler, doppler_bins, cpi_timestamps = \
            generate_range_doppler(data, num_pulses_per_cpi=args.num_pulses_per_cpi,
                                   keep_clutter=args.keep_clutter)

        # Set up formatters
        range_formatter = ValueFormatter(data['range_bins'])
        pulse_formatter = ValueFormatter((data['timestamps'] - data['timestamps'][0]) / 1000)
        doppler_formatter = ValueFormatter(doppler_bins)

        # Plot RTI
        rti_fig = plt.figure()
        rti_ax = rti_fig.add_subplot(111)
        h_img = rti_ax.imshow(20 * np.log10(np.abs(data['scan_data'])))
        rti_ax.set_aspect('auto')
        rti_ax.set_title('Range-Time Intensity')
        rti_ax.set_xlabel('Range (m) [Range Bin Number]')
        rti_ax.set_ylabel('Time Elapsed (s) [Pulse Number]')
        rti_ax.xaxis.set_major_formatter(range_formatter)
        rti_ax.yaxis.set_major_formatter(pulse_formatter)
        cbar = rti_fig.colorbar(h_img)
        cbar.ax.set_ylabel('dB')

        # Try to display to screen if available otherwise save to file
        plt.ion()
        plt.show()
        if args.unpacked_data_filename:
            rti_filename = Path(args.unpacked_data_filename)
            rti_filename = rti_filename.with_name('{0}.png'.format(rti_filename.stem))
        else:
            rti_filename = Path('RTI.png')
        rti_filename = deconflict_file(rti_filename)
        plt.savefig(rti_filename)
        input('Press [enter] to continue.')

        # Plot range-Doppler
        if args.unpacked_data_filename:
            range_doppler_filename = Path(args.unpacked_data_filename)
            range_doppler_filename = range_doppler_filename.with_name('{0}.gif'.format(
                    range_doppler_filename.stem))
        else:
            range_doppler_filename = Path('range_doppler.gif')
        range_doppler_filename = deconflict_file(range_doppler_filename)
        rd_plot = RangeDopplerPlot(range_doppler, cpi_timestamps, range_formatter,
                                   doppler_formatter)
        rd_plot.start()
        rd_plot.save(range_doppler_filename)
        input('Press [enter] to continue.')

    # Return unpacked scan_data
    return data

if __name__ == "__main__":
    """Standard Python alias for command line execution."""
    main(sys.argv[1:])
