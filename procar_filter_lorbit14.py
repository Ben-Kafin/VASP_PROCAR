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
          filter_types : A list of element symbols.
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
          - Line 2: Scaling factor.
          - Lines 3-5: Lattice vectors.
          - Line 6: Element symbols (space separated).
          - Line 7: Number of atoms for each element (space separated).
          
        Returns:
          A dictionary mapping atom numbers (starting at 1) to element symbols.
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
        
        For each state, the code reads the individual orbital contributions for each ion
        from the ion data lines. It saves these contributions in a dictionary under the
        key "contrib_by_ion" and also builds an orbital composition matrix ("orbital_matrix")
        for the state. The number of columns is determined dynamically from the header line.
        
        For lorbit=14, a header line will be present (usually beginning with "ion" and
        containing "tot"). In such a case:
          - The header (ignoring the first column "ion") gives the orbital names.
          - Then for each subsequent line, the first token is the ion index and the remaining tokens
            (up to the expected number) are read as contributions.
          - When a line is encountered whose first token starts with "tot", the values (from token 2
            to the expected columns) are converted to floats and appended as an extra row to the
            orbital matrix. These tot values are then stored under the key "orbital_totals"
            in the state (and are not saved under "contrib_by_ion").
          - After processing the tot row, the inner loop breaks (skipping the second, charge data block).
        
        If no header line is detected, it defaults to 11 tokens, with the default orbital headers:
        ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2", "tot"].
        (That produces a 10-column orbital matrix.)
        """
        if not os.path.exists(self.procar_path):
            print("PROCAR file not found in", self.directory)
            return
    
        with open(self.procar_path, "r") as f:
            lines = f.readlines()
    
        self.states = []
        i = 0
        current_kpoint = None
        current_kweight = 1.0  # default weight if not found
    
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
    
            # Look for a k-point header.
            k_match = re.search(r"k[- ]?point\s+(\d+).*weight\s*=\s*([-\d.Ee]+)", line, re.IGNORECASE)
            if k_match:
                current_kpoint = int(k_match.group(1))
                current_kweight = float(k_match.group(2))
                i += 1
                # Process bands for this k-point until the next k-point header.
                while i < len(lines) and not re.search(r"k[- ]?point", lines[i], re.IGNORECASE):
                    band_line = lines[i].strip()
                    # Look for a band header.
                    b_match = re.search(
                        r"band\s+(\d+)(.*?)#\s+energy\s+([-\d.Ee]+)\s+#\s+occ\.\s+([-\d.Ee]+)",
                        band_line, re.IGNORECASE
                    )
                    if b_match:
                        band_index = int(b_match.group(1))
                        extra_text = b_match.group(2).strip()
                        band_name = extra_text if extra_text else f"band {band_index}"
                        energy = float(b_match.group(3))
                        occ = float(b_match.group(4))
    
                        # Determine the number of ions.
                        N = max(self.atom_type_map.keys()) if self.atom_type_map else 0
    
                        # These variables will be set from a header line if present.
                        expected_cols = None
                        header_parsed = False
                        orbital_matrix = None
    
                        # Create a state dictionary.
                        state = {
                            "kpoint": current_kpoint,
                            "kweight": current_kweight,
                            "band": band_index,
                            "band_name": band_name,
                            "energy": energy,
                            "occ": occ,
                            "contrib_by_ion": {}
                        }
                        i += 1
                        # Process ion contribution lines until the next band or k-point header.
                        while i < len(lines) and not re.search(r"^(band\s+|k[- ]?point)", lines[i], re.IGNORECASE):
                            ion_line = lines[i].strip()
                            # If the header line is not yet parsed, check for it.
                            if not header_parsed:
                                tokens = ion_line.split()
                                if tokens and tokens[0].lower().startswith("ion") and ("tot" in ion_line.lower()):
                                    expected_cols = len(tokens)
                                    # Use the header from the PROCAR file.
                                    state["orbital_headers"] = tokens[1:]  # Ignore the "ion" column.
                                    orbital_matrix = np.zeros((N, len(state["orbital_headers"])))
                                    header_parsed = True
                                    i += 1
                                    continue
                                else:
                                    # If no header is encountered, use the default of 11 tokens.
                                    expected_cols = 11
                                    state["orbital_headers"] = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2", "tot"]
                                    orbital_matrix = np.zeros((N, len(state["orbital_headers"])))
                                    header_parsed = True
                                    # Do not increment i; process the current line as data.
                            tokens = ion_line.split()
                            # Skip empty lines.
                            if not tokens:
                                i += 1
                                continue
                            # Process the tot row: keep it in the band matrix.
                            if tokens[0].lower().startswith("tot"):
                                try:
                                    tot_values = [float(x) for x in tokens[1:expected_cols]]
                                except Exception:
                                    i += 1
                                    continue
                                orbital_matrix = np.vstack([orbital_matrix, tot_values])
                                state["orbital_totals"] = dict(zip(state["orbital_headers"], tot_values))
                                i += 1
                                # Break out to skip the charge data block.
                                break
                            # Otherwise, process a standard ion contribution line.
                            if len(tokens) < expected_cols:
                                i += 1
                                continue
                            elif len(tokens) > expected_cols:
                                tokens = tokens[:expected_cols]
                            try:
                                ion_index = int(tokens[0])
                            except ValueError:
                                i += 1
                                continue
                            try:
                                values = [float(x) for x in tokens[1:]]
                            except ValueError:
                                i += 1
                                continue
                            contrib = {header: val for header, val in zip(state["orbital_headers"], values)}
                            state["contrib_by_ion"][ion_index] = contrib
                            if ion_index <= N:
                                orbital_matrix[ion_index - 1, :] = values
                            i += 1
                        state["orbital_matrix"] = orbital_matrix
                        self.states.append(state)
                    else:
                        i += 1
                continue
            else:
                i += 1
    
        print("Parsed", len(self.states), "states from PROCAR.")

    def build_state_types(self):
        """
        (Optional) Adds a new key 'types' to each state.
        Iterates over each ion in a state and uses the atom_type_map to determine the element;
        only ions with nonzero total contribution are included.
        """
        for state in self.states:
            types_set = set()
            tot_key = state.get("orbital_headers", ["tot"])[-1]
            for ion_index, contrib in state.get("contrib_by_ion", {}).items():
                if contrib.get(tot_key, 0.0) != 0:
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
            state_types = {x.upper() for x in state.get("types", [])}
            if state_types and state_types.issubset(allowed_set):
                continue
            filtered_states.append(state)
        self.states = filtered_states
        print("After filtering, {} states remain.".format(len(self.states)))

    def plot_states(self):
        """
        Plots each state's aggregated total contribution (using the last column of orbital_matrix)
        per ion as vertical delta lines for visualization.
        """
        if not self.states:
            print("No states available for plotting; run parse_procar() first.")
            return
        plt.figure(figsize=(10, 6))
        for state in self.states:
            base_x = state["energy"] - self.fermi_level
            tot_array = state.get("orbital_matrix", np.array([]))[:, -1]
            for value in tot_array:
                plt.vlines(base_x, 0, value, colors="black", linewidth=1)
        plt.xlabel("Energy (eV) relative to Fermi level")
        plt.ylabel("Total contribution")
        plt.title("PROCAR States: Aggregated total contributions per ion")
        plt.grid(True)
        plt.show()

    def combine_kpoint_bands(self):
        """
        Combines states from different k-points that correspond to the same band.
        A weighted average (using k-point weights) of the orbital_matrix, energy, and occupancy is computed.
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
                "band_name": states[0].get("band_name", f"band {band}"),
                "energy": avg_energy,
                "occ": avg_occ,
                "orbital_matrix": combined_matrix,
                "total_weight": total_weight
            })
        
        combined_states = sorted(combined_states, key=lambda s: s["energy"])
        self.states_combined = combined_states
        return combined_states

    def export_combined_bands(self, out_filename="combined_composition_matrices.txt"):
        """
        Exports the combined composition matrix for each band into a tab-delimited file.
        The file is written into the provided directory.
        """
        combined_states = self.combine_kpoint_bands()
        out_lines = []
        for new_index, state in enumerate(combined_states, start=1):
            M = state["orbital_matrix"]
            norm_val = np.linalg.norm(M)
            norm_M = M / norm_val if norm_val != 0 else M
            out_lines.append(f"Band {new_index}\tOriginalBand: {state['band']}\tEnergy: {state['energy']:.4f} eV")
            num_mapped = len(self.atom_type_map)
            n_rows = state["orbital_matrix"].shape[0]
            mapped_indices = sorted(self.atom_type_map.keys())
            restricted_rows = []
            for idx in mapped_indices:
                if idx <= n_rows:
                    restricted_rows.append(state["orbital_matrix"][idx-1])
                else:
                    restricted_rows.append(np.zeros(M.shape[1]))
            restricted_matrix = np.array(restricted_rows)
            fro_norm = np.linalg.norm(restricted_matrix)
            norm_restricted = restricted_matrix / fro_norm if fro_norm != 0 else restricted_matrix
            for row in norm_restricted:
                row_str = "\t".join(f"{val:.4e}" for val in row)
                out_lines.append(row_str)
            out_lines.append("")
        # Build the full file path in the provided directory.
        full_out_filename = os.path.join(self.directory, out_filename)
        with open(full_out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined band composition matrices to {full_out_filename}")

    def run(self):
        self.parse_procar()
        self.build_state_types()
        self.filter_states()
        self.plot_states()
        self.export_combined_bands()

# ----- Run the Code -----
if __name__ == "__main__":
    directory = 'dir'
    fermi_level = -1.9849  # Example value; adjust as needed.
    filter_types = []
    
    plotter = ProcarStatePlotter(directory, fermi_level, filter_types)
    plotter.run()
