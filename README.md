This folder contains a couple sets of code for parseing and analyzing VASP PROCAR files. 

Specifically, the files "procar_filter",and the two Band_matching files are designed for VASP calculations using lorbit = 10 which writes the PROCAR and DOSCAR with only the angular-spin resolved s, p, and d orbitals.
  procar_filter parses a PROCAR file and removes requested atoms from individual bands. 
    It also can use the kpoint weights for multiple kpoint calculations to make a weighted sum of each kpoint for a given band.
      results in a singular data block (orbital matrix) for a single band that contains the atom indicies and the orbital population for each atom as well as the total population of each atom.
    Band matchers will take two input directories, on of a lone molecule in a vacuum, and one of the same molecule adsorbed onto a surface. 
      Will align deepest energy state (not involved with bonding, energy shouldn't change upon molecular adsorption) so comparing energies between two different VASP calculations makes sense.
        shifts all lone molecule states to align deepest energy level. Then shift both full system and aligned lone molecular system so that full system's fermi energy is at 0eV.
      It will find a band from the lone molecule system that matches a band from the full system using a dot product of the normalized orbital matricies of each band.
      Each full system band has a best-match lone molecule band.
      Then it will plot both the lone molecular PROCAR and the full system PROCAR with individual states that matched having the same color.
      Will also write several output txt files that contain the individual band information for each system as well as a list of all matching states and the energy shift.


Procar_filter_lorbit14 and the Molecular_Orbital_projection_matcheing_cmap_interactive code are designed to work with VASP calculation using lorbit = 14.
  lorbit = 14 writes the PROCAR and DOSCAR with magnetic spin resolved orbitals (s, px, py, px, dxy, dxy, dyz, dxz, dx2-y2, and dz2).
  procar_filter_lorbit14 works exactly as the og procar_filter, but dynamically uses data headers and column numbers to parse any number of orbitals in a PROCAR.
  Molecular Orbital Projection Matching cmap interactive is the same logic and comes from the same orignal files as the band matchers (and works with og procar filter if import line is changed).
    Also uses a divergent color map for the plotting so states that have big changes are identifiable and all simple states have unique colors.
    Uses mplcursors (must "pip install mplcursors") so the final plot will tell you the full system band that the cursor is hovering over with the following information:
      Full band#, lone molecular band#, Energy Shift (eV).
    Will write two additional txt files.
      one that is every match ordered by the absolute value of their energy shift.
      One that for each lone molecular band will take a weighted sum of the energy shifts of every full system  to get a single numerical value to give a general  behavior of the lone molecular state upon adsorption.
        weights are the total population (y value) of each state normalized by the original lone molecular state's total poulation.
  
