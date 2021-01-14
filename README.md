# BGCArgo_BCGyre

A new an improved version of BGCArgoPython with better and more organized code :)

## Things To Do
- [X] Sort Argo floats by geographic regions
- [ ] Interpolate data to same pressure levels
- [ ] Look at oxygen over time in different regions along isopycnals?

## Python Scripts
- ***GetArgoFiles.py***: Uses synthetic profile index to get BGC Argo floats in a certain geographic region (0-80ºW, 40-80ºN)
  - Outputs DacWMO, WMO, Index text files, and floats sorted by dac (use for downloading floats from ftp site)
- ***SortFloatsByType.py***: Visually inspect float trajectories and categorize them by 'type'
  - Types: Boundary current, Labrador Sea, Irminger Sea, Other, or N/A
  - Output: Sorted_WMO_<TYPE>Float.txt and Sorted_DACWMO_<TYPE>Float.txt
- ***BGCArgoGasFlux.py***: Calculates the air-sea oxygen gas flux for BGC Argo floats
  - Interpolates Argo float data to match ERA5 data timesteps, uses nearest neighbor to get ERA5 data and calculate air-sea flux at each point
  - Input: Sorted_DACWMO_<TYPE>Float.txt files and ERA5 wind and SLP data
  - Output: Figures of surface variables (temperature, salinity, oxygen) float trajectory, and air-sea oxygen flux
- ***RandomFxns.py***: A collection of random functions
  - DetermineAdjusted: determines if adjusted/not-adjust Argo float data is used; if adjusted data is present, these data are used
    - Output: adjusted_flag; if flag = 1 adjusted data were used
  - Other functions to add
    - [ ] Calculate MLD
