import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from procar_filter import ProcarStatePlotter

class CombinedBandMatcher:
    def __init__(self, simple_dir, full_dir, max_distance=0.01):
        """
        Initializes the CombinedBandMatcher.
        
        Parameters:
          simple_dir   : Directory containing the PROCAR and POSCAR files for the simple system.
          full_dir     : Directory containing the PROCAR and POSCAR files for the full system.
          max_distance : Maximum allowed distance (in POSCAR coordinate units) for matching atoms.
          
        The Fermi energy is read automatically from the DOSCAR file in the full system directory.
        """
        self.simple_dir = simple_dir
        self.full_dir = full_dir
        self.max_distance = max_distance
        
        # Get the Fermi level from DOSCAR in the full system directory.
        self.fermi_level = self.get_fermi_level_from_doscar(self.full_dir)
        if self.fermi_level is None:
            raise ValueError("Fermi energy could not be determined from DOSCAR in the full system directory!")
        print(f"Using Fermi energy from DOSCAR: {self.fermi_level:.4f} eV")
        
        # Create ProcarStatePlotter instances using the determined Fermi energy.
        self.simple_plotter = ProcarStatePlotter(simple_dir, self.fermi_level)
        self.full_plotter   = ProcarStatePlotter(full_dir, self.fermi_level)
        
        # Combined band states (one per unique band, after combining the k-point data).
        self.simple_combined = []
        self.full_combined = []
        
        # Atom mapping from the POSCAR files: { simple_atom_index: full_atom_index }.
        self.atom_mapping = {}

    def get_fermi_level_from_doscar(self, directory):
        """
        Reads the DOSCAR file in the specified directory and returns the Fermi energy.
        It assumes that the Fermi energy is the fourth number (index 3) on the sixth line (index 5).
        For example, if the sixth line is:
        
          "   -5.4321  0.0000  0.0000  -12.34567890  0.00123 ..."
        
        then the Fermi energy will be -12.34567890.
        """
        doscar_path = os.path.join(directory, "DOSCAR")
        with open(doscar_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 6:
                raise ValueError("DOSCAR file does not have at least 6 lines.")
            # Get the sixth line and split it into tokens.
            tokens = lines[5].strip().split()
            # The fourth token (index 3) should hold the Fermi energy.
            ef = float(tokens[3])
        return ef


    def parse_poscar_from_dir(self, directory):
        """
        Reads the POSCAR file in the given directory and returns a dictionary mapping atom indices
        to a dict with keys "element" and "coord" (a numpy array of [x, y, z]).
        Assumes VASP 5+ format with optional Selective Dynamics.
        """
        poscar_path = None
        for fname in os.listdir(directory):
            if fname.lower() == "poscar":
                poscar_path = os.path.join(directory, fname)
                break
        if not poscar_path:
            return {}
        try:
            with open(poscar_path, "r") as f:
                lines = [line.rstrip() for line in f if line.strip()]
            if len(lines) < 7:
                return {}
            tokens = lines[1].split()
            if len(tokens) == 1:
                lattice_start = 2
            else:
                lattice_start = 1
            symbols = lines[lattice_start+3].split()
            counts = [int(x) for x in lines[lattice_start+4].split()]
            total_atoms = sum(counts)
            spec_line = lines[lattice_start+5].strip().lower()
            if spec_line.startswith("selective"):
                pos_start = lattice_start+7
            else:
                if "direct" in spec_line or spec_line[0]=='d' or "cartesian" in spec_line or spec_line[0]=='c':
                    pos_start = lattice_start+6
                else:
                    pos_start = lattice_start+5
            coords = []
            for i in range(pos_start, pos_start+total_atoms):
                pos_tokens = lines[i].split()[:3]
                coord = np.array([float(x) for x in pos_tokens])
                coords.append(coord)
            mapping = {}
            atom_number = 1
            pos_idx = 0
            for sym, count in zip(symbols, counts):
                for _ in range(count):
                    mapping[atom_number] = {"element": sym, "coord": coords[pos_idx]}
                    pos_idx += 1
                    atom_number += 1
            return mapping
        except Exception:
            return {}

    def get_atom_mapping(self):
        """
        Constructs a one-to-one mapping between atoms in the simple and full systems using their POSCAR files.
        For each simple atom, only full system atoms with the same element are considered; the closest (by Euclidean distance)
        is chosen if within max_distance. The mapping is stored as: { simple_atom_index: full_atom_index, ... }.
        """
        simple_atoms = self.parse_poscar_from_dir(self.simple_dir)
        full_atoms = self.parse_poscar_from_dir(self.full_dir)
        mapping = {}
        tol = self.max_distance
        for s_idx, s_data in simple_atoms.items():
            s_elem = s_data["element"]
            s_coord = s_data["coord"]
            best_dist = float("inf")
            best_f_idx = None
            for f_idx, f_data in full_atoms.items():
                if f_data["element"] != s_elem:
                    continue
                dist = np.linalg.norm(s_coord - f_data["coord"])
                if dist < best_dist:
                    best_dist = dist
                    best_f_idx = f_idx
            if best_f_idx is not None and best_dist <= tol:
                mapping[s_idx] = best_f_idx
        self.atom_mapping = mapping

    def load_combined_states(self):
        """
        Loads PROCAR states for each system (via ProcarStatePlotter) and combines the k-point data
        (weighted by the k-point weights) for each unique band.
        The combined states are stored in self.simple_combined and self.full_combined.
        Also exports the combined composition matrices to text files in their respective directories.
        """
        self.simple_plotter.parse_procar()
        self.full_plotter.parse_procar()
        self.simple_combined = self.simple_plotter.combine_kpoint_bands()
        self.full_combined = self.full_plotter.combine_kpoint_bands()
        simple_out = os.path.join(self.simple_dir, "simple_combined_composition_matrices.txt")
        full_out = os.path.join(self.full_dir, "full_combined_composition_matrices.txt")
        self.export_combined_bands(self.simple_combined, simple_out)
        self.export_combined_bands(self.full_combined, full_out)

    def assign_sequential_indices(self):
        """
        Sorts and assigns new sequential indices (stored as 'seq') to the combined states for both systems based on increasing energy.
        """
        self.simple_combined = sorted(self.simple_combined, key=lambda s: s["energy"])
        for i, state in enumerate(self.simple_combined, start=1):
            state["seq"] = i
        self.full_combined = sorted(self.full_combined, key=lambda s: s["energy"])
        for i, state in enumerate(self.full_combined, start=1):
            state["seq"] = i

    def adjust_simple_combined_band_energies(self):
        """
        Shifts the energies of the simple combined bands by a constant so that the lowest-energy simple band aligns with
        the lowest-energy full band.
        Returns the computed energy shift.
        """
        if not self.full_combined or not self.simple_combined:
            return 0.0
        sorted_full = sorted(self.full_combined, key=lambda s: s["energy"])
        sorted_simple = sorted(self.simple_combined, key=lambda s: s["energy"])
        energy_shift = sorted_full[0]["energy"] - sorted_simple[0]["energy"]
        for state in self.simple_combined:
            state["energy"] += energy_shift
        return energy_shift

    def shift_energies_to_fermi(self):
        """
        Subtracts the full system's Fermi energy (from DOSCAR) from every combined state's energy,
        so that the full system's Fermi energy is at 0 eV in both systems.
        """
        for state in self.simple_combined:
            state["energy"] -= self.fermi_level
        for state in self.full_combined:
            state["energy"] -= self.fermi_level

    def export_combined_bands(self, combined_states, out_filename):
        """
        Exports the combined composition matrices into a tab-delimited text file.
        The states are assigned new sequential indices (stored in 'seq') and sorted by energy.
        For each band, a header is printed that shows the new sequential index, original band number, and the shifted energy.
        Then the 2-D normalized composition matrix (restricted to mapped atoms) is output.
        The file is saved in the same directory as the PROCAR file.
        """
        self.assign_sequential_indices()
        out_lines = []
        for state in combined_states:
            M = self.restrict_matrix(state, system="simple", flatten=False)
            norm_val = np.linalg.norm(M)
            norm_M = M / norm_val if norm_val != 0 else M
            out_lines.append(f"Band {state['seq']}\tOriginalBand: {state['band']}\tEnergy: {state['energy']:.4f} eV")
            for row in norm_M:
                out_lines.append("\t".join(f"{val:.4e}" for val in row))
            out_lines.append("")
        with open(out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined composition matrices to {out_filename}")

    def restrict_matrix(self, state, system="simple", flatten=True):
        """
        Restricts the combined state's orbital_matrix (of shape (N,4)) to only the rows corresponding
        to atoms present in the simple system.
        
        For system "simple": uses the simple POSCAR atom indices.
        For system "full": uses the atom mapping (i.e. for each simple atom, the corresponding row in the full system).
        Returns a flattened 1-D vector if flatten is True, or a 2-D array otherwise.
        """
        keys_sorted = sorted(self.atom_mapping.keys())
        if "orbital_matrix" not in state:
            return None
        if system == "simple":
            arr = np.array([state["orbital_matrix"][idx-1] for idx in keys_sorted if idx-1 < state["orbital_matrix"].shape[0]])
        elif system == "full":
            arr = np.array([state["orbital_matrix"][self.atom_mapping[idx]-1] for idx in keys_sorted if self.atom_mapping[idx]-1 < state["orbital_matrix"].shape[0]])
        else:
            raise ValueError("system must be 'simple' or 'full'")
        return arr.flatten() if flatten else arr

    def get_removed_matrix(self, state):
        """
        For a given full system combined state, returns a tuple:
         (removed_matrix, list_of_removed_atom_numbers)
        where removed_matrix holds the rows in the full system's orbital_matrix that do not correspond to any mapped simple atom.
        """
        if "orbital_matrix" not in state:
            return None
        full_matrix = state["orbital_matrix"]
        N = full_matrix.shape[0]
        kept = set([v for v in self.atom_mapping.values() if v <= N])
        all_indices = set(range(1, N+1))
        removed = sorted(list(all_indices - kept))
        if len(removed) == 0:
            return None
        removed_matrix = full_matrix[np.array(removed)-1, :]
        return removed_matrix, removed

    def get_removed_orbital_info(self, state):
        """
        Computes a summary for the removed atoms in a full system state.
        For each removed atom, produces a string including its full system atom number and its orbital contributions (s, p, d, tot).
        Then sorts the atoms by the 'tot' column (descending) and returns a semicolon-separated string of the top three atoms.
        """
        result = self.get_removed_matrix(state)
        if result is None:
            return "None"
        R, removed_list = result
        info_lines = []
        for i, atom_num in enumerate(removed_list):
            s_val, p_val, d_val, tot_val = R[i]
            info_lines.append((atom_num, tot_val, f"Atom {atom_num}: s={s_val:.4e}, p={p_val:.4e}, d={d_val:.4e}, tot={tot_val:.4e}"))
        info_lines_sorted = sorted(info_lines, key=lambda x: x[1], reverse=True)
        top_count = min(3, len(info_lines_sorted))
        top_info = "; ".join(info[2] for info in info_lines_sorted[:top_count])
        return top_info

    def export_removed_bands(self):
        """
        Exports the removed composition matrices (for extra full system atoms not mapped to the simple system)
        into a separate file in the full system directory.
        The format is equivalent to that of the combined composition matrix export.
        """
        out_filename = os.path.join(self.full_dir, "full_removed_composition_matrices.txt")
        out_lines = []
        sorted_states = sorted(self.full_combined, key=lambda s: s["energy"])
        for state in sorted_states:
            result = self.get_removed_matrix(state)
            if result is None:
                continue
            R, removed_list = result
            norm_val = np.linalg.norm(R)
            norm_R = R / norm_val if norm_val != 0 else R
            out_lines.append(f"Band {state.get('seq', state['band'])}\tOriginalBand: {state['band']}\tEnergy: {state['energy']:.4f} eV")
            out_lines.append(f"Removed Atoms: {removed_list}")
            for row in norm_R:
                out_lines.append("\t".join(f"{val:.4e}" for val in row))
            out_lines.append("")
        with open(out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported removed composition matrices to {out_filename}")

    def match_combined_bands(self):
        """
        For each full system combined band, restricts its composition matrix (using full mapping) to the mapped atoms,
        normalizes the flattened vector, and computes its inner product with every normalized simple system combined band's
        restricted vector (using simple mapping). The simple band with the highest absolute overlap is chosen as the match.
        Also computes the energy difference (full energy - simple energy) and extracts removed orbital info.
        Returns a list of matches as tuples:
          (full_state, best_simple_state, overlap, energy_difference, removed_orb_info),
        using the new sequential indices ("seq") for both systems.
        """
        self.get_atom_mapping()
        self.load_combined_states()
        self.assign_sequential_indices()
        energy_shift = self.adjust_simple_combined_band_energies()
        print(f"Applied energy shift of {energy_shift:.4f} eV to simple combined bands.")
        self.shift_energies_to_fermi()
        matches = []
        for full_state in self.full_combined:
            v_full = self.restrict_matrix(full_state, system="full", flatten=True)
            if v_full is None or np.linalg.norm(v_full) == 0:
                continue
            v_full_norm = v_full / np.linalg.norm(v_full)
            best_overlap = 0.0
            best_simple = None
            for simple_state in self.simple_combined:
                v_simple = self.restrict_matrix(simple_state, system="simple", flatten=True)
                if v_simple is None or np.linalg.norm(v_simple) == 0:
                    continue
                v_simple_norm = v_simple / np.linalg.norm(v_simple)
                overlap = np.dot(v_full_norm, v_simple_norm)
                if abs(overlap) > abs(best_overlap):
                    best_overlap = overlap
                    best_simple = simple_state
            if best_simple is not None:
                energy_diff = full_state["energy"] - best_simple["energy"]
            else:
                energy_diff = None
            removed_info = self.get_removed_orbital_info(full_state)
            matches.append((full_state, best_simple, best_overlap, energy_diff, removed_info))
        matches.sort(key=lambda m: ((m[1]["seq"] if m[1] is not None else 1e6), m[0]["seq"]))
        return matches

    def final_plot(self, matches):
        """
        Produces the final figure with two subplots:
          - Top subplot: simple system combined bands are displayed as vertical delta lines (with inverted y-axis).
          - Bottom subplot: full system combined bands are displayed as vertical delta lines, colored by the matched simple state.
        The x-axis is energy (with the full system's Fermi level at 0 eV) and the y-axis is total composition (sum over the "tot" column).
        A dotted black vertical line is also drawn at x=0 in both subplots.
        The figure is saved to the full system directory.
        """
        simple_sorted = sorted(self.simple_combined, key=lambda s: s["energy"])
        full_sorted = sorted(self.full_combined, key=lambda s: s["energy"])
        
        def total_composition(state, system):
            M = self.restrict_matrix(state, system=system, flatten=False)
            return np.sum(M[:, 3])
        
        simple_energy = [s["energy"] for s in simple_sorted]
        simple_total = [total_composition(s, "simple") for s in simple_sorted]
        full_energy = [s["energy"] for s in full_sorted]
        full_total = [total_composition(s, "full") for s in full_sorted]
        
        cmap = plt.get_cmap("tab10")
        simple_colors = { s["seq"]: cmap(i % 10) for i, s in enumerate(simple_sorted, start=1) }
        
        full_colors = []
        for state in full_sorted:
            match = next((m for m in matches if m[0]["seq"] == state["seq"]), None)
            if match is not None and match[1] is not None:
                color = simple_colors.get(match[1]["seq"], "black")
            else:
                color = "gray"
            full_colors.append(color)
        
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        
        # Plot simple bands.
        for energy, total, s in zip(simple_energy, simple_total, simple_sorted):
            color = simple_colors.get(s["seq"], "black")
            ax_top.vlines(energy, 0, total, colors=color, lw=1)
        ax_top.axvline(x=0, color='black', linestyle='dotted')
        ax_top.set_ylabel("Total Composition")
        ax_top.set_title("Simple Combined Bands (Upside-Down)")
        ax_top.invert_yaxis()
        
        # Plot full bands.
        for energy, total, s, color in zip(full_energy, full_total, full_sorted, full_colors):
            ax_bottom.vlines(energy, 0, total, colors=color, lw=1)
        ax_bottom.axvline(x=0, color='black', linestyle='dotted')
        ax_bottom.set_ylabel("Total Composition")
        ax_bottom.set_xlabel("Energy (eV)")
        ax_bottom.set_title("Full Combined Bands")
        
        fig.suptitle("Combined Band Matching: Energy vs Total Composition", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_fig = os.path.join(self.full_dir, "combined_band_matching_figure.png")
        plt.savefig(out_fig, dpi=300)
        plt.show()
        print(f"Final band matching figure saved to {out_fig}")

    def export_matches(self, matches):
        """
        Exports the matching results to a tab-delimited text file in the full system directory.
        The output includes: FullBand(seq), FullEnergy, SimpleBand(seq), SimpleEnergy, Overlap, EnergyDifference, and RemovedOrbitals.
        """
        out_filename = os.path.join(self.full_dir, "combined_band_matching_results.txt")
        out_lines = []
        header = "FullBand(seq)\tFullEnergy(eV)\tSimpleBand(seq)\tSimpleEnergy(eV)\tOverlap\tEnergyDifference(eV)\tRemovedOrbitals"
        out_lines.append(header)
        for full_state, simple_state, overlap, energy_diff, removed_info in matches:
            if simple_state is None:
                line = f"{full_state.get('seq', full_state['band'])}\t{full_state['energy']:.4f}\tNoMatch\t-\t-\t-\t{removed_info}"
            else:
                line = f"{full_state.get('seq', full_state['band'])}\t{full_state['energy']:.4f}\t{simple_state.get('seq', simple_state['band'])}\t{simple_state['energy']:.4f}\t{overlap:.4f}\t{energy_diff:.4f}\t{removed_info}"
            out_lines.append(line)
        with open(out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined band matching results to {out_filename}")

    def run(self):
        matches = self.match_combined_bands()
        self.export_combined_bands(self.simple_combined, os.path.join(self.simple_dir, "simple_combined_composition_matrices.txt"))
        self.export_combined_bands(self.full_combined, os.path.join(self.full_dir, "full_combined_composition_matrices.txt"))
        self.export_removed_bands()
        self.export_matches(matches)
        self.final_plot(matches)
        return matches

if __name__ == "__main__":
    simple_dir = 'd' 
    full_dir = 'd'
    # No fermi_level input needed since it is extracted from DOSCAR.
    
    matcher = CombinedBandMatcher(simple_dir, full_dir, max_distance=0.01)
    matcher.run()
