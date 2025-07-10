import os
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from collections import defaultdict
from procar_filter_lorbit14 import ProcarStatePlotter #keep in same folder as procar_filter_lorbit14

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
        """
        doscar_path = os.path.join(directory, "DOSCAR")
        with open(doscar_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 6:
                raise ValueError("DOSCAR file does not have at least 6 lines.")
            tokens = lines[5].strip().split()
            ef = float(tokens[3])
        return ef

    def parse_poscar_from_dir(self, directory):
        """
        Reads the POSCAR file in the given directory and returns a dictionary mapping atom indices
        to a dict with keys "element" and "coord" (a numpy array of [x, y, z]), with coordinates converted
        to Cartesian units if they are provided in Direct (fractional) form.
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
                scaling_factor = float(tokens[0])
                lattice_start = 2
            else:
                scaling_factor = 1.0
                lattice_start = 1
    
            lattice_vectors = []
            for i in range(lattice_start, lattice_start + 3):
                vec = np.array([float(x) for x in lines[i].split()])
                lattice_vectors.append(vec)
            lattice_vectors = np.array(lattice_vectors) * scaling_factor
    
            symbols = lines[lattice_start + 3].split()
            counts = [int(x) for x in lines[lattice_start + 4].split()]
            total_atoms = sum(counts)
    
            spec_line = lines[lattice_start + 5].strip().lower()
            if spec_line.startswith("selective"):
                coordinate_mode = lines[lattice_start + 6].strip().lower()
                coord_line_index = lattice_start + 7
            else:
                coordinate_mode = spec_line
                coord_line_index = lattice_start + 6
    
            is_direct = True
            if "cartesian" in coordinate_mode or coordinate_mode.startswith("c"):
                is_direct = False
    
            coords = []
            for i in range(coord_line_index, coord_line_index + total_atoms):
                pos_tokens = lines[i].split()[:3]
                frac_coord = np.array([float(x) for x in pos_tokens])
                if is_direct:
                    cart_coord = np.dot(frac_coord, lattice_vectors)
                    coords.append(cart_coord)
                else:
                    coords.append(frac_coord)
    
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
        Loads PROCAR states for each system and combines the k-point data for each unique band.
        Exports the combined composition matrices to text files in their respective directories.
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
        Sorts and assigns new sequential indices (stored as 'seq') based on increasing energy.
        """
        self.simple_combined = sorted(self.simple_combined, key=lambda s: s["energy"])
        for i, state in enumerate(self.simple_combined, start=1):
            state["seq"] = i
        self.full_combined = sorted(self.full_combined, key=lambda s: s["energy"])
        for i, state in enumerate(self.full_combined, start=1):
            state["seq"] = i

    def adjust_simple_combined_band_energies(self):
        """
        Shifts the energies of the simple combined bands so that the lowest-energy simple band aligns with
        the lowest-energy full band. Returns the energy shift.
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
        Subtracts the full system's Fermi energy from every combined state's energy.
        """
        for state in self.simple_combined:
            state["energy"] -= self.fermi_level
        for state in self.full_combined:
            state["energy"] -= self.fermi_level

    def export_combined_bands(self, combined_states, out_filename):
        """
        Exports the combined composition matrix for each band into a tab-delimited file.
        The file is saved in the provided directory.
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
        full_out_filename = os.path.join(self.full_dir, out_filename)
        with open(full_out_filename, "w") as fout:
            for line in out_lines:
                fout.write(line + "\n")
        print(f"Exported combined composition matrices to {full_out_filename}")

    def restrict_matrix(self, state, system="simple", flatten=True):
        """
        Restricts the combined state's orbital_matrix (of shape (N, number_of_orbitals)) to only the rows corresponding
        to atoms present in the simple system.
        
        For system "simple": uses the simple POSCAR atom indices.
        For system "full": uses the atom mapping from simple to full.
        Returns a flattened 1-D vector if flatten is True, otherwise returns a 2-D array.
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
        For each removed full-system atom, produces a summary string of its orbital contributions.
        Sorts by the 'tot' column (descending) and returns a semicolon-separated string of the top three atoms.
        """
        result = self.get_removed_matrix(state)
        if result is None:
            return "None"
        R, removed_list = result
        info_lines = []
        for i, atom_num in enumerate(removed_list):
            # Use the last column as the total.
            tot_val = R[i, -1]
            # Adjust if your orbitals include more than three components; here we only show tot.
            info_lines.append((atom_num, tot_val, f"Atom {atom_num}: tot={tot_val:.4e}"))
        info_lines_sorted = sorted(info_lines, key=lambda x: x[1], reverse=True)
        top_count = min(3, len(info_lines_sorted))
        top_info = "; ".join(info[2] for info in info_lines_sorted[:top_count])
        return top_info

    def match_combined_bands(self):
        """
        For each full system combined band, restricts its composition matrix (using full mapping) to the mapped atoms,
        normalizes the resulting flattened vector, and computes its inner product with every normalized simple system's
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




    def final_plot(self,
                   matches,
                   energy_range=None,
                   center_seq=None,
                   cmap_name="coolwarm",
                   power=1.0):
        """
        Top subplot: unchanged simple‐band plotting (matched=colored solid, unmatched=black dashed).
    
        Bottom subplot: first draws full bands whose simple match is off-range (black/gray),
        then in-range matched bands in ascending |energy_shift| order so the most-shifted sit on top.
    
        center_seq → colormap midpoint, cmap_name → any Matplotlib cmap,
        power <1 → quick ramp, >1 → plateau near center.
        """
        # 1) Sort and prepare colormap
        all_simple = sorted(self.simple_combined, key=lambda s: s["energy"])
        all_full   = sorted(self.full_combined,   key=lambda s: s["energy"])
        N = len(all_simple)
        cmap = plt.get_cmap(cmap_name)
    
        # 2) Compute pivot and half_span
        if center_seq and 1 <= center_seq <= N:
            pivot = (center_seq - 1) / (N - 1)
        else:
            pivot = 0.5
        half_span = max(pivot, 1 - pivot)
    
        # 3) Build seq → color mapping
        simple_colors = {}
        for idx, state in enumerate(all_simple):
            raw   = idx / (N - 1)
            dx    = (raw - pivot) / half_span
            dx    = np.clip(dx, -1, 1)
            dx_pw = np.sign(dx) * (abs(dx)**power)
            t     = 0.5 + 0.5 * dx_pw
            t     = np.clip(t, 0.0, 1.0)
            simple_colors[state["seq"]] = cmap(t)
    
        # 4) Filter by energy_range if given
        if energy_range:
            emin, emax = energy_range
            simple_plot = [s for s in all_simple if emin <= s["energy"] <= emax]
            full_plot   = [f for f in all_full   if emin <= f["energy"] <= emax]
        else:
            simple_plot, full_plot = all_simple, all_full
    
        # 5) Find matched simple‐seqs
        matched_seqs = {m[1]["seq"] for m in matches if m[1]}
    
        # helper to sum the last column of orbital_matrix
        def total_comp(state, system):
            M = self.restrict_matrix(state, system=system, flatten=False)
            return np.sum(M[:, -1]) if M is not None else 0
    
        # 6) Set up figure
        fig, (ax_top, ax_bot) = plt.subplots(2,1,sharex=True,figsize=(8,6))
    
        # — Top: simple‐band plotting (original logic) —
        for s in simple_plot:
            e = s["energy"]
            t = total_comp(s, "simple")
            if s["seq"] in matched_seqs:
                col, lw, ls = simple_colors[s["seq"]], 2, "solid"
            else:
                col, lw, ls = "black",            1, "dashed"
            ax_top.vlines(e, 0, t, colors=col, lw=lw, linestyle=ls)
    
        ax_top.axvline(0, color="k", linestyle="dotted")
        ax_top.set_title("Lone NHC Orbital Composed DOS")
        ax_top.set_ylabel("DOS")
    
        # — Bottom: full‐system in two stages —
        # prepare to collect full‐system artists for mplcursors
        full_artists = []
    
        # 7a) Group1: unmatched or off-range matched
        group1 = []
        # 7b) Group2: in-range matched
        group2 = []
        for f in full_plot:
            m = next((m for m in matches if m[0]["seq"] == f["seq"]), None)
            if m and m[1]:
                s = m[1]
                # off-range simple → group1
                if energy_range and not (emin <= s["energy"] <= emax):
                    group1.append((f, m))
                else:
                    group2.append((f, m))
            else:
                # no match at all
                group1.append((f, m))
    
        # 8) Sort group2 by ascending |energy_shift|
        group2.sort(key=lambda fm: abs(fm[1][3]) if fm[1] else 0.0)
    
        # 9) Plot group1 (draw first)
        for full_state, m in group1:
            e = full_state["energy"]
            t = total_comp(full_state, "full")
            if m and m[1]:
                col = "black"   # off-range
            else:
                col = "gray"    # no match
            lc = ax_bot.vlines(e, 0, t, colors=col, lw=2)
            # attach band info for hover
            lc.band_info = {
                "full_seq":   full_state["seq"],
                "simple_seq": (m[1]["seq"] if m and m[1] else None),
                "E_shift":    (m[3]      if m else None)
            }
            full_artists.append(lc)
    
        # 10) Plot group2 (draw atop)
        for full_state, m in group2:
            e = full_state["energy"]
            t = total_comp(full_state, "full")
            col = simple_colors[m[1]["seq"]]
            lc = ax_bot.vlines(e, 0, t, colors=col, lw=2)
            lc.band_info = {
                "full_seq":   full_state["seq"],
                "simple_seq": m[1]["seq"],
                "E_shift":    m[3]
            }
            full_artists.append(lc)
    
        ax_bot.axvline(0, color="k", linestyle="dotted")
        ax_bot.set_title("Balbot fcc NHC Orbital Composed DOS")
        ax_bot.set_ylabel("DOS")
        ax_bot.set_xlabel("Energy (eV)")
    
        if energy_range:
            ax_top.set_xlim(energy_range)
            ax_bot.set_xlim(energy_range)
    
        fig.suptitle(
            "Molecular Orbital Mapping",
            fontsize=14
        )
        plt.tight_layout(rect=[0,0,1,0.95])
    
        # ——————————————
        # mplcursors hover popups
        # ——————————————
        cursor = mplcursors.cursor(full_artists, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            info = sel.artist.band_info
            sel.annotation.set_text(
                f"Full band: {info['full_seq']}\n"
                f"Simple band: {info['simple_seq']}\n"
                f"ΔE = {info['E_shift']:.3f} eV"
            )
            sel.annotation.get_bbox_patch().set(alpha=0.8)
    
        out_fig = os.path.join(self.full_dir, "combined_band_matching_figure.png")
        plt.savefig(out_fig, dpi=300)
        plt.show()
        print(f"Final figure saved to {out_fig}")
        
    def run(self,
            energy_range=None,
            center_seq=None,
            cmap_name="seismic",
            power=1.0):
        matches = self.match_combined_bands()

        self.export_combined_bands(
            self.simple_combined,
            os.path.join(self.simple_dir, "simple_combined_composition_matrices.txt")
        )
        self.export_combined_bands(
            self.full_combined,
            os.path.join(self.full_dir,   "full_combined_composition_matrices.txt")
        )
        self.export_matches(matches)
        self.export_matches_by_shift(matches)

        # <-- call the new exporter here -->
        self.export_simple_weighted_shifts(matches)

        self.final_plot(
            matches,
            energy_range=energy_range,
            center_seq=center_seq,
            cmap_name=cmap_name,
            power=power
        )
        return matches

    def export_matches(self, matches):
        """
        Exports the matching results to a tab-delimited text file in the full system directory.
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

    def export_matches_by_shift(self, matches):
        """
        Writes all matches sorted by descending absolute energy shift
        into full_dir/combined_band_matching_by_shift.txt
        """
        # sort: largest |energy_diff| first; push None diffs to the end
        sorted_by_shift = sorted(
            matches,
            key=lambda m: abs(m[3]) if (m[3] is not None) else -1.0,
            reverse=True
        )
    
        out_path = os.path.join(self.full_dir,
                                "combined_band_matching_by_shift.txt")
        header = ("FullSeq\tFullEnergy(eV)\tSimpleSeq\tSimpleEnergy(eV)"
                  "\tOverlap\tEnergyDiff(eV)\tRemovedOrbitals")
    
        with open(out_path, "w") as fout:
            fout.write(header + "\n")
            for full_s, simple_s, ov, ed, rem in sorted_by_shift:
                if simple_s is None:
                    line = (f"{full_s['seq']}\t{full_s['energy']:.4f}"
                            f"\tNoMatch\t-\t-\t-\t{rem}")
                else:
                    line = (f"{full_s['seq']}\t{full_s['energy']:.4f}"
                            f"\t{simple_s['seq']}\t{simple_s['energy']:.4f}"
                            f"\t{ov:.4f}\t{ed:.4f}\t{rem}")
                fout.write(line + "\n")
        print(f"Wrote matches by shift to {out_path}")
        
    def export_simple_weighted_shifts(self, matches):
        """
        For each simple‐system band, compute a weighted sum of full‐system energy shifts:
          weight = (total_population_full_band) / (total_population_simple_band)
        Writes results to full_dir/simple_states_weighted_shifts.txt
        """
        # 1) group all full‐state matches by simple_seq
        grouped = defaultdict(list)
        for full_s, simple_s, overlap, ediff, removed in matches:
            if simple_s is None or ediff is None:
                continue
            grouped[simple_s["seq"]].append((full_s, ediff))

        # 2) output path
        out_path = os.path.join(self.full_dir, "simple_states_weighted_shifts.txt")

        # 3) write header + one line per simple band in seq order
        with open(out_path, "w") as fout:
            fout.write("SimpleSeq\tSimpleEnergy(eV)\tWeightedShift(eV)\n")
            for s in sorted(self.simple_combined, key=lambda x: x["seq"]):
                seq    = s["seq"]
                energy = s["energy"]
                # total pop of the simple band (sum of tot column)
                M_simple = self.restrict_matrix(s, system="simple", flatten=False)
                pop_simple = np.sum(M_simple[:, -1]) if M_simple is not None else 0.0

                # accumulate weighted shift
                weighted_shift = 0.0
                if pop_simple > 0 and seq in grouped:
                    for full_s, ed in grouped[seq]:
                        M_full = self.restrict_matrix(full_s, system="full", flatten=False)
                        pop_full = np.sum(M_full[:, -1]) if M_full is not None else 0.0
                        weight = pop_full / pop_simple
                        weighted_shift += weight * ed

                fout.write(f"{seq}\t{energy:.4f}\t{weighted_shift:.4f}\n")

        print(f"Wrote simple‐state weighted shifts to {out_path}")




if __name__ == "__main__":
    simple_dir = 'C:/dir1'
    full_dir   = 'C:/dir2'
    
    matcher = CombinedBandMatcher(simple_dir, full_dir, max_distance=0.1)
    matcher.run(energy_range=(-4.32, 8.5), center_seq=40,cmap_name="coolwarm",power=0.25
)
