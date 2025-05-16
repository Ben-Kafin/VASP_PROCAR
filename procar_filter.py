# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:57:33 2025

@author: Benjamin Kafin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:15:52 2025

@author: Benjamin Kafin
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class ProcarStatePlotter:
    def __init__(self, directory="path/to/directory", fermi_level=0.0, filter_types=None):
        """
        Initializes the state plotter.
        
        Parameters:
          directory    : Folder containing the PROCAR and POSCAR files.
                         Update this with your actual folder path.
          fermi_level  : Fermi level (in eV); energies will be plotted as (energy - fermi_level).
          filter_types : A list of element symbols. (Not used in the current matching but available for filtering later.)
        """
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.fermi_level = fermi_level
        self.filter_types = filter_types  # For optional filtering.
        self.states = []  # Will store state dictionaries from PROCAR.
        self.states_combined = []  # Will store combined states (one per unique band).
        # Build a mapping from ion (atom) index to its element symbol.
        self.atom_type_map = self.parse_poscar()  

    def parse_poscar(self):
        """
        Parses the POSCAR file to build a mapping from atom (ion) indices to element symbols.
        Assumes the VASP 5+ format:
          - Line 1: Comment/title.
          - Line 2: Scaling factor (or, if it has 3 tokens, assume scale = 1 and lattice starts immediately).
          - Lines 3-5: Lattice vectors.
          - Line 6: Element symbols (space separated).
          - Line 7: Number of atoms for each element (space separated).
          
        Returns:
          A dictionary mapping atom numbers (starting at 1) to element symbols.
          For example: {1: "Au", 2: "N", 3: "N", 4: "C", ...}
        """
        mapping = {}
        if not os.path.exists(self.poscar_path):
            print("POSCAR file not found in", self.directory)
            return mapping
        with open(self.poscar_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        if len(lines) < 7:
            print("POSCAR file is too short; cannot determine atom types.")
            return mapping
        elements = lines[5].split()
        try:
            counts = [int(x) for x in lines[6].split()]
        except Exception as e:
            print("Error parsing atom counts in POSCAR:", e)
            return mapping
        current_index = 1
        for i, elem in enumerate(elements):
            num = counts[i] if i < len(counts) else 0
            for _ in range(num):
                mapping[current_index] = elem  # Save as is.
                current_index += 1
        return mapping

    def parse_procar(self):
        """
        Parses the new-format PROCAR file to extract every state.
        A state is defined by its k-point and band.
        
        For each state, the code reads the individual orbital contributions for each ion from the ion data lines.
        It saves these contributions (s, p, d, tot) for each ion index in a dictionary under the key "contrib_by_ion".
        
        Additionally, it builds an orbital composition matrix (shape: (N, 4)), where N is number of atoms (from POSCAR).
        The state dictionary includes:
            'kpoint', 'band', 'energy', 'occ', 'kweight',
            'contrib_by_ion': { ion_index: {"s":..., "p":..., "d":..., "tot":...}, ... }
            'orbital_matrix': A NumPy array of shape (N,4) (rows ordered by atom index).
        """
        if not os.path.exists(self.procar_path):
            print("PROCAR file not found in", self.directory)
            return

        with open(self.procar_path, "r") as f:
            lines = f.readlines()

        self.states = []
        current_kpoint = None
        current_kweight = 1.0  # default weight if not found.
        current_band = None
        current_energy = None
        current_occ = None
        current_contrib_ion = {}  # Contributions for the current band.

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Look for a k-point line.
            # Example k-point line: "k-point     1 :  0.00000000 0.00000000 0.00000000     weight = 1.00000000"
            k_match = re.match(r"k-point\s+(\d+)\s*:", line)
            if k_match:
                current_kpoint = int(k_match.group(1))
                # Attempt to extract weight using re.search.
                weight_match = re.search(r"weight\s*=\s*([-\d.Ee]+)", line)
                if weight_match:
                    current_kweight = float(weight_match.group(1))
                else:
                    current_kweight = 1.0
                i += 1
                continue

            # Look for a band line.
            b_match = re.match(r"band\s+(\d+)\s+#\s+energy\s+([-\d.Ee]+)\s+#\s+occ\.\s+([\d.Ee]+)", line)
            if b_match:
                # Save the previous state if one was in progress.
                if current_band is not None:
                    # Build the orbital matrix.
                    if self.atom_type_map:
                        N = max(self.atom_type_map.keys())
                    else:
                        N = max(current_contrib_ion.keys()) if current_contrib_ion else 0
                    orb_matrix = np.zeros((N, 4))
                    for ion_index, orb_dict in current_contrib_ion.items():
                        orb_matrix[ion_index-1, 0] = orb_dict.get("s", 0.0)
                        orb_matrix[ion_index-1, 1] = orb_dict.get("p", 0.0)
                        orb_matrix[ion_index-1, 2] = orb_dict.get("d", 0.0)
                        orb_matrix[ion_index-1, 3] = orb_dict.get("tot", 0.0)
                    state = {
                        "kpoint": current_kpoint,
                        "kweight": current_kweight,
                        "band": current_band,
                        "energy": current_energy,
                        "occ": current_occ,
                        "contrib_by_ion": current_contrib_ion.copy(),
                        "orbital_matrix": orb_matrix
                    }
                    self.states.append(state)
                # Start a new band.
                current_band = int(b_match.group(1))
                current_energy = float(b_match.group(2))
                current_occ = float(b_match.group(3))
                current_contrib_ion = {}  # Reset contributions for the new band.
                i += 1
                continue

            # Skip the ion header line.
            if line.lower().startswith("ion") and "tot" in line.lower():
                i += 1
                continue

            # Process an ion data line.
            ion_match = re.match(r"^\s*(\d+)\s+([-\d.Ee]+)\s+([-\d.Ee]+)\s+([-\d.Ee]+)\s+([-\d.Ee]+)", line)
            if ion_match:
                ion_index = int(ion_match.group(1))
                s_val = float(ion_match.group(2))
                p_val = float(ion_match.group(3))
                d_val = float(ion_match.group(4))
                tot_val = float(ion_match.group(5))
                current_contrib_ion[ion_index] = {"s": s_val, "p": p_val, "d": d_val, "tot": tot_val}
                i += 1
                continue

            i += 1

        # Save the last state.
        if current_band is not None:
            if self.atom_type_map:
                N = max(self.atom_type_map.keys())
            else:
                N = max(current_contrib_ion.keys()) if current_contrib_ion else 0
            orb_matrix = np.zeros((N, 4))
            for ion_index, orb_dict in current_contrib_ion.items():
                orb_matrix[ion_index-1, 0] = orb_dict.get("s", 0.0)
                orb_matrix[ion_index-1, 1] = orb_dict.get("p", 0.0)
                orb_matrix[ion_index-1, 2] = orb_dict.get("d", 0.0)
                orb_matrix[ion_index-1, 3] = orb_dict.get("tot", 0.0)
            state = {
                "kpoint": current_kpoint,
                "kweight": current_kweight,
                "band": current_band,
                "energy": current_energy,
                "occ": current_occ,
                "contrib_by_ion": current_contrib_ion.copy(),
                "orbital_matrix": orb_matrix
            }
            self.states.append(state)

        print("Parsed", len(self.states), "states from PROCAR.")

    def build_state_types(self):
        """
        (Optional) Adds a new key 'types' to each state.
        Iterates over each ion in a state and uses the atom_type_map to determine the element;
        only ions with nonzero total contribution are included.
        """
        for state in self.states:
            types_set = set()
            for ion_index, contrib in state.get("contrib_by_ion", {}).items():
                if contrib.get("tot", 0.0) != 0:
                    elem = self.atom_type_map.get(ion_index, "Unknown")
                    types_set.add(elem)
            state["types"] = sorted(list(types_set))

    def filter_states(self):
        """
        (Optional) Filters out states that consist exclusively of atom types listed in self.filter_types.
        """
        if self.filter_types is None:
            return
        allowed_set = set(x.upper() for x in self.filter_types)
        filtered_states = []
        for state in self.states:
            state_types = set(x.upper() for x in state.get("types", []))
            if state_types and state_types.issubset(allowed_set):
                continue
            filtered_states.append(state)
        self.states = filtered_states
        print("After filtering, {} states remain.".format(len(self.states)))

    def plot_states(self):
        """
        Plots each state's aggregated total ("tot") contribution per ion as vertical delta lines.
        This is mainly for visualization.
        """
        if not self.states:
            print("No states available for plotting; run parse_procar() first.")
            return
        plt.figure(figsize=(10, 6))
        for state in self.states:
            base_x = state["energy"] - self.fermi_level
            tot_array = state.get("orbital_matrix", np.array([]))[:, 3]
            for value in tot_array:
                plt.vlines(base_x, 0, value, colors="black", linewidth=1)
        plt.xlabel("Energy (eV) relative to Fermi level")
        plt.ylabel("Total contribution")
        plt.title("PROCAR States: Aggregated tot contributions per ion")
        plt.grid(True)
        plt.show()

    def combine_kpoint_bands(self):
        """
        Combines states from different k-points that correspond to the same band (i.e. have the same 'band' value).
        For each unique band, it computes a weighted average (using k-point weights) of:
           - the orbital_matrix,
           - the energy, and
           - the occupancy.
        The combined state for each band is returned as a new dictionary.
        """
        band_groups = defaultdict(list)
        for state in self.states:
            band_groups[state["band"]].append(state)
        combined_states = []
        for band, states in band_groups.items():
            total_weight = sum(s["kweight"] for s in states)
            combined_matrix = sum(s["orbital_matrix"] * s["kweight"] for s in states) / total_weight
            avg_energy = sum(s["energy"] * s["kweight"] for s in states) / total_weight
            avg_occ = sum(s["occ"] * s["kweight"] for s in states) / total_weight
            combined_states.append({
                "band": band,
                "energy": avg_energy,
                "occ": avg_occ,
                "orbital_matrix": combined_matrix,
                "total_weight": total_weight
            })
        # Sort the combined states in increasing order of energy.
        combined_states = sorted(combined_states, key=lambda s: s["energy"])
        self.states_combined = combined_states
        return combined_states

    def export_combined_bands(self, out_filename="combined_composition_matrices.txt"):
        """
        Exports the final (weighted combined) composition matrix for each band into a tab-delimited text file.
        The combined states (from all k-points) are sorted in increasing order of energy and
        re-indexed sequentially. For each band, a header line (with new sequential index, original band, and energy)
        is printed followed by the 2-D normalized composition matrix (rows correspond to the mapped atoms,
        columns are [s, p, d, tot]).
        """
        combined_states = self.combine_kpoint_bands()
        out_lines = []
        # Sort by energy and re-index sequentially.
        for new_index, state in enumerate(combined_states, start=1):
            M = state["orbital_matrix"]
            norm_val = np.linalg.norm(M)
            if norm_val != 0:
                norm_M = M / norm_val
            else:
                norm_M = M
            out_lines.append(f"Band {new_index}\tOriginalBand: {state['band']}\tEnergy: {state['energy']:.4f} eV")
            # Reshape M to 2-D: number of mapped atoms x 4.
            # Get the number of mapped atoms from the atom mapping.
            num_mapped = len(self.atom_type_map)  # Assuming all atoms are in the mapping.
            # However, note that the orbital matrix was created with shape (N, 4) where N is maximum atom index.
            # It may be safer to simply get the shape of the original orbital matrix.
            n_rows = state["orbital_matrix"].shape[0]
            # Now, we want to output only the rows corresponding to the mapped atoms.
            # We assume that the rows to output are in the same order as sorted(self.atom_type_map.keys())
            mapped_indices = sorted(self.atom_type_map.keys())
            # Build a 2-D array for the restricted matrix:
            restricted_rows = []
            for idx in mapped_indices:
                if idx <= n_rows:
                    restricted_rows.append(state["orbital_matrix"][idx-1])
                else:
                    restricted_rows.append(np.zeros(4))
            restricted_matrix = np.array(restricted_rows)
            # Normalize the entire restricted matrix using Frobenius norm.
            fro_norm = np.linalg.norm(restricted_matrix)
            if fro_norm != 0:
                norm_restricted = restricted_matrix / fro_norm
            else:
                norm_restricted = restricted_matrix
            for row in norm_restricted:
                row_str = "\t".join(f"{val:.4e}" for val in row)
                out_lines.append(row_str)
            out_lines.append("")
        with open(out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined band composition matrices to {out_filename}")

    def run(self):
        self.parse_procar()
        self.build_state_types()
        self.filter_states()
        self.plot_states()
        # Combine states from different k-points for each band and export composition matrices.
        self.export_combined_bands()
        
# ----- Run the Code -----
if __name__ == "__main__":
    # Update the directory path and fermi level as needed.
    directory = "C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/"
    fermi_level = 0  # Example value; adjust as needed.
    filter_types = ["Au"]
    
    plotter = ProcarStatePlotter(directory, fermi_level, filter_types)
    plotter.run()